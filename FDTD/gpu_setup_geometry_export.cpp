/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#include "gpu_setup_geometry_export.h"
#include "gpu_operator_setup.h"
#include "CSPrimBox.h"
#include "CSPrimSphere.h"
#include "CSPrimCylinder.h"
#include "CSPrimSphericalShell.h"
#include "CSPrimCylindricalShell.h"
#include "CSPrimLinPoly.h"
#include "CSPrimPolygon.h"
#include "CSPrimRotPoly.h"
#include "CSPrimMultiBox.h"
#include "CSPrimWire.h"
#include "CSPrimPoint.h"
#include "CSPrimPolyhedron.h"
#include "CSPrimPolyhedronReader.h"
#include "CSPrimUserDefined.h"
#include "CSProperties.h"
#include "CSTransform.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <typeinfo>
#include "gpu_setup_type_key.h"
#include <map>
#include <mutex>

namespace
{
void Require(bool valid, const char* reason)
{
	if (!valid) throw GPU_SetupUnavailable(reason);
}

// CSXCAD inverts all four rows, then applies the first three without dividing
// by w. Testing only the upper 3x3 incorrectly rejects valid input matrices.
double Determinant(const double* m)
{
	double result=0;
	for (int column=0;column<4;++column)
	{
		double a[9]; int offset=0;
		for (int row=1;row<4;++row) for (int c=0;c<4;++c) if (c!=column) a[offset++]=m[4*row+c];
		const double minor=a[0]*(a[4]*a[8]-a[5]*a[7])-a[1]*(a[3]*a[8]-a[5]*a[6])+a[2]*(a[3]*a[7]-a[4]*a[6]);
		result+=(column%2 ? -1 : 1)*m[column]*minor;
	}
	return result;
}
}

void GPU_SetupExportInverse(const double matrix[16], double output[12])
{
	for (int k=0;k<16;++k) Require(std::isfinite(matrix[k]),"non-finite input transform");
	const double determinant=Determinant(matrix);
	Require(std::isfinite(determinant) && determinant!=0,"singular input transform");
	CSTransform inverse;
	inverse.SetMatrix(matrix,false); inverse.Invert();
	std::copy(inverse.GetMatrix(),inverse.GetMatrix()+12,output);
	for (int k=0;k<12;++k) Require(std::isfinite(output[k]),"non-finite inverse transform");
}

static GPU_SetupPrimitive DescribeBuiltin(CSPrimitives& prim,const std::type_info& semantic,std::vector<double>& data,GPU_SetupExpressionBank* expressions)
{
	GPU_SetupPrimitive d={};
	d.version=2; d.id=prim.GetID(); d.property_id=prim.GetProperty()->GetUniqueID();
	d.dimension=prim.GetDimension();
	d.inverse[0]=d.inverse[5]=d.inverse[10]=1;
	d.forward[0]=d.forward[5]=d.forward[10]=1;
	d.coordinate_system=prim.GetCoordinateSystem()==CYLINDRICAL ? 1 : 0;
	d.input_coordinate_system=prim.GetCoordInputType()==CYLINDRICAL ? 1 : 0;
	d.has_transform=prim.HasTransform();
	if (prim.GetCoordinateSystem()==UNDEFINED_CS) d.coordinate_system=d.input_coordinate_system;
	if (prim.HasTransform())
	{
		CSTransform inverse(prim.GetTransform());
		for (int k=0;k<16;++k) Require(std::isfinite(inverse.GetMatrix()[k]),"non-finite primitive transform");
		const double determinant=Determinant(inverse.GetMatrix());
		Require(std::isfinite(determinant) && determinant!=0,"singular primitive transform");
		std::copy(inverse.GetMatrix(),inverse.GetMatrix()+12,d.forward);
		inverse.Invert();
		std::copy(inverse.GetMatrix(),inverse.GetMatrix()+12,d.inverse);
	}
	d.candidate_bounds=prim.GetBoundBox(d.bounds) && !prim.HasTransform() &&
	                   (prim.GetBoundBoxCoordSystem()==UNDEFINED_CS || prim.GetBoundBoxCoordSystem()==prim.GetCoordInputType());
	const double* start=NULL; const double* stop=NULL;
	if (semantic==typeid(CSPrimBox))
	{
		CSPrimBox& box=dynamic_cast<CSPrimBox&>(prim); d.shape=GPU_SetupShape::Box;
		const CoordinateSystem cs=d.coordinate_system ? CYLINDRICAL : CARTESIAN;
		start=box.GetStartCoord()->GetCoords(cs); stop=box.GetStopCoord()->GetCoords(cs);
	}
	else if (semantic==typeid(CSPrimSphere) || semantic==typeid(CSPrimSphericalShell))
	{
		CSPrimSphere& sphere=dynamic_cast<CSPrimSphere&>(prim); d.shape=GPU_SetupShape::Sphere;
		start=stop=sphere.GetCenter()->GetCartesianCoords(); d.radius=sphere.GetRadius();
		if (semantic==typeid(CSPrimSphericalShell))
		{ d.shape=GPU_SetupShape::SphericalShell; d.shell_width=dynamic_cast<CSPrimSphericalShell&>(prim).GetShellWidth(); }
	}
	else if (semantic==typeid(CSPrimCylinder) || semantic==typeid(CSPrimCylindricalShell))
	{
		CSPrimCylinder& cylinder=dynamic_cast<CSPrimCylinder&>(prim); d.shape=GPU_SetupShape::Cylinder;
		start=cylinder.GetAxisStartCoord()->GetCartesianCoords(); stop=cylinder.GetAxisStopCoord()->GetCartesianCoords();
		d.radius=cylinder.GetRadius();
		Require(start[0]!=stop[0] || start[1]!=stop[1] || start[2]!=stop[2],"zero-length cylinder");
		if (semantic==typeid(CSPrimCylindricalShell))
		{ d.shape=GPU_SetupShape::CylindricalShell; d.shell_width=dynamic_cast<CSPrimCylindricalShell&>(prim).GetShellWidth(); }
	}
	else if (semantic==typeid(CSPrimPolygon) || semantic==typeid(CSPrimLinPoly) || semantic==typeid(CSPrimRotPoly))
	{
		CSPrimPolygon& poly=dynamic_cast<CSPrimPolygon&>(prim); d.shape=GPU_SetupShape::Polygon;
		if (semantic==typeid(CSPrimLinPoly)) d.geometry_flags|=1u;
		d.start[0]=poly.GetElevation();
		d.stop[0]=semantic==typeid(CSPrimLinPoly) ? dynamic_cast<CSPrimLinPoly&>(prim).GetLength() : 0;
		Require(poly.GetNormDir()>=0 && poly.GetNormDir()<3,"invalid polygon normal");
		Require(data.size()<=UINT32_MAX && poly.GetQtyCoords()<=(UINT32_MAX-data.size())/2,"polygon vertex index range exceeded");
		d.normal=static_cast<uint32_t>(poly.GetNormDir()); d.offset=static_cast<uint32_t>(data.size()); d.count=static_cast<uint32_t>(poly.GetQtyCoords());
		for (uint32_t v=0;v<d.count;++v) { data.push_back(poly.GetCoord(2*v)); data.push_back(poly.GetCoord(2*v+1)); }
		if (semantic==typeid(CSPrimRotPoly))
		{
			CSPrimRotPoly& rot=dynamic_cast<CSPrimRotPoly&>(prim); d.shape=GPU_SetupShape::RotPoly;
			Require(rot.GetRotAxisDir()>=0 && rot.GetRotAxisDir()<3,"invalid rotational polygon axis");
			d.axis=static_cast<uint32_t>(rot.GetRotAxisDir()); d.angles[0]=rot.GetAngle(0); d.angles[1]=rot.GetAngle(1);
			const double period=2*acos(-1.0);
			if (d.angles[0]>d.angles[1]) d.angles[1]+=period;
			if (d.angles[0]>period) { d.angles[0]-=period; d.angles[1]-=period; }
			if (d.angles[0]<0) { d.angles[0]+=period; d.angles[1]+=period; }
		}
	}
	else if (semantic==typeid(CSPrimMultiBox))
	{
		CSPrimMultiBox& boxes=dynamic_cast<CSPrimMultiBox&>(prim); d.shape=GPU_SetupShape::MultiBox;
		Require(data.size()<=UINT32_MAX && boxes.GetQtyBoxes()<=(UINT32_MAX-data.size())/6,"MultiBox index range exceeded");
		d.offset=static_cast<uint32_t>(data.size()); d.count=boxes.GetQtyBoxes();
		for (uint32_t v=0;v<6*d.count;++v) data.push_back(boxes.GetCoord(v));
	}
	else if (semantic==typeid(CSPrimWire) || semantic==typeid(CSPrimCurve))
	{
		CSPrimCurve& wire=dynamic_cast<CSPrimCurve&>(prim); d.shape=GPU_SetupShape::Curve;
		if (semantic==typeid(CSPrimWire)) { d.shape=GPU_SetupShape::Wire; d.radius=dynamic_cast<CSPrimWire&>(prim).GetWireRadius(); }
		Require(data.size()<=UINT32_MAX && wire.GetNumberOfPoints()<=(UINT32_MAX-data.size())/3,"wire vertex index range exceeded");
		d.offset=static_cast<uint32_t>(data.size()); d.count=static_cast<uint32_t>(wire.GetNumberOfPoints());
		for (uint32_t v=0;v<d.count;++v)
		{
			double point[3]; Require(wire.GetPoint(v,point,CARTESIAN,false),"invalid wire point");
			data.insert(data.end(),point,point+3);
		}
		Require(data.size()<=UINT32_MAX && d.count<=(UINT32_MAX-data.size())/3,"wire raw coordinate range exceeded");
		d.bounds_offset=static_cast<uint32_t>(data.size());d.bounds_count=d.count;
		for (uint32_t v=0;v<d.count;++v) {double point[3];Require(wire.GetPoint(v,point),"invalid raw wire point");data.insert(data.end(),point,point+3);}
	}
	else if (semantic==typeid(CSPrimPoint))
	{
		d.shape=GPU_SetupShape::Point;for (int n=0;n<3;++n) d.start[n]=dynamic_cast<CSPrimPoint&>(prim).GetCoord(n);
	}
	else if (semantic==typeid(CSPrimUserDefined))
		throw GPU_SetupUnavailable("UserDefined geometry requires CPU operator construction");
	else if (semantic==typeid(CSPrimPolyhedron) || semantic==typeid(CSPrimPolyhedronReader))
		throw GPU_SetupUnavailable("polyhedron geometry requires CPU operator construction");
	else throw GPU_SetupUnavailable("no device geometry adapter for "+prim.GetTypeName());
	if (start) { std::copy(start,start+3,d.start); std::copy(stop,stop+3,d.stop); }
	for (double value:d.inverse) Require(std::isfinite(value),"non-finite inverse transform");
	for (double value:d.start) Require(std::isfinite(value),"non-finite geometry coordinate");
	for (double value:d.stop) Require(std::isfinite(value),"non-finite geometry coordinate");
	for (double value:d.bounds) Require(std::isfinite(value),"non-finite geometry bounds");
	Require(std::isfinite(d.radius) && std::isfinite(d.shell_width) && std::isfinite(d.angles[0]) && std::isfinite(d.angles[1]),"non-finite geometry parameter");
	return d;
}

namespace
{
const std::type_info* const builtin_types[]={&typeid(CSPrimBox),&typeid(CSPrimSphere),&typeid(CSPrimSphericalShell),
	&typeid(CSPrimCylinder),&typeid(CSPrimCylindricalShell),&typeid(CSPrimPolygon),&typeid(CSPrimLinPoly),&typeid(CSPrimRotPoly),
	&typeid(CSPrimMultiBox),&typeid(CSPrimWire),&typeid(CSPrimCurve),&typeid(CSPrimPoint),&typeid(CSPrimUserDefined),
	&typeid(CSPrimPolyhedron),&typeid(CSPrimPolyhedronReader)};

bool Builtin(const std::type_info& type)
{
	for (const auto* candidate:builtin_types) if (type==*candidate) return true;
	return false;
}

struct PrimitiveAdapter {std::string name;GPU_PrimitivePrepare prepare;GPU_PrimitiveExport export_data;};
std::mutex& AdapterMutex() {static std::mutex mutex;return mutex;}
std::map<std::string,PrimitiveAdapter>& Adapters()
{
	static std::map<std::string,PrimitiveAdapter> adapters=[]() {
		std::map<std::string,PrimitiveAdapter> result;
		for (const auto* type:builtin_types)
			result.emplace(GPU_SetupTypeKey(*type),PrimitiveAdapter{"CSXCAD built-in",[](CSPrimitives&,std::string&) {return true;},
			    [type](CSPrimitives& prim,std::vector<double>& data,GPU_SetupExpressionBank* expressions) {return GPU_DescribePrimitiveAs(prim,*type,data,expressions);}});
		return result;
	}();
	return adapters;
}
}

void GPU_RegisterPrimitiveAdapter(const std::type_info& type,const std::string& name,GPU_PrimitivePrepare prepare,GPU_PrimitiveExport export_data)
{
	if (name.empty() || !prepare || !export_data) throw std::invalid_argument("primitive adapter needs a name, preparation check and exporter");
	std::lock_guard<std::mutex> lock(AdapterMutex());
	if (Adapters().count(GPU_SetupTypeKey(type))) throw std::invalid_argument("primitive adapter is already registered");
	Adapters().emplace(GPU_SetupTypeKey(type),PrimitiveAdapter{name,std::move(prepare),std::move(export_data)});
}

void GPU_UnregisterPrimitiveAdapter(const std::type_info& type)
{
	if (Builtin(type)) throw std::invalid_argument("built-in primitive adapters cannot be removed");
	std::lock_guard<std::mutex> lock(AdapterMutex());Adapters().erase(GPU_SetupTypeKey(type));
}

GPU_SetupPrimitive GPU_DescribePrimitiveAs(CSPrimitives& prim,const std::type_info& semantic,std::vector<double>& data,GPU_SetupExpressionBank* expressions)
{
	if (!Builtin(semantic)) throw GPU_SetupUnavailable("unknown built-in geometry adapter");
	try {return DescribeBuiltin(prim,semantic,data,expressions);}
	catch (const std::bad_cast&) {throw GPU_SetupUnavailable("geometry adapter requires a compatible CSXCAD base class");}
}

GPU_SetupPrimitive GPU_DescribePrimitive(CSPrimitives& prim,std::vector<double>& data,GPU_SetupExpressionBank* expressions)
{
	PrimitiveAdapter adapter;
	{
		std::lock_guard<std::mutex> lock(AdapterMutex());const auto found=Adapters().find(GPU_SetupTypeKey(typeid(prim)));
		if (found==Adapters().end()) throw GPU_SetupUnavailable("no device geometry adapter for "+prim.GetTypeName());
		adapter=found->second;
	}
	std::string reason;
	if (!adapter.prepare(prim,reason)) throw GPU_SetupUnavailable(adapter.name+": "+reason);
	const auto result=adapter.export_data(prim,data,expressions);
	Require(result.version==2 && result.id==prim.GetID() && result.property_id==prim.GetProperty()->GetUniqueID(),"geometry adapter changed descriptor version or stable identity");
	Require(static_cast<uint32_t>(result.shape)<=static_cast<uint32_t>(GPU_SetupShape::UserDefined),"geometry adapter returned an unsupported device query");
	const uint32_t stride=result.shape==GPU_SetupShape::Polygon || result.shape==GPU_SetupShape::RotPoly ? 2 :
	                      result.shape==GPU_SetupShape::MultiBox ? 6 : result.shape==GPU_SetupShape::Polyhedron ? 9 :
	                      result.shape==GPU_SetupShape::Curve || result.shape==GPU_SetupShape::Wire ? 3 : 0;
	Require(result.offset<=data.size() && (!stride || result.count<=(data.size()-result.offset)/stride),"geometry adapter buffer range exceeded");
	Require(result.bounds_offset<=data.size() && result.bounds_count<=(data.size()-result.bounds_offset)/3,"geometry adapter bounds range exceeded");
	return result;
}
