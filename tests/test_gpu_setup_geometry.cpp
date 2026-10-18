/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#include "FDTD/gpu_setup_geometry.h"
#include "FDTD/gpu_setup_geometry_export.h"
#include "FDTD/gpu_setup_memory.h"
#include "FDTD/gpu_setup_constant.h"
#include "FDTD/gpu_setup_candidates.h"
#include "CSPropMaterial.h"
#include "ContinuousStructure.h"
#include "CSPrimBox.h"
#include "CSPrimSphere.h"
#include "CSPrimCylinder.h"
#include "CSPrimSphericalShell.h"
#include "CSPrimCylindricalShell.h"
#include "CSPrimPolygon.h"
#include "CSPrimLinPoly.h"
#include "CSPrimRotPoly.h"
#include "CSPrimMultiBox.h"
#include "CSPrimWire.h"
#include "CSPrimPoint.h"
#include "CSPrimPolyhedron.h"
#include "CSPrimPolyhedronReader.h"
#include "CSPrimUserDefined.h"
#include "CSTransform.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <functional>
#include <chrono>
#include <cstdio>
#include <array>
#include <map>

#ifdef OPENEMS_DEVICE_GEOMETRY_TEST
void GPU_TestGeometrySamples(const GPU_SetupPrimitive&, const std::vector<double>&,
                             const GPU_SetupExpressionBank&, const std::vector<double>&, std::vector<unsigned char>&);
void GPU_TestPropertyIDs(const std::vector<GPU_SetupPrimitive>&,const std::vector<double>&,const std::vector<double>&,std::vector<uint32_t>&);
#endif

namespace
{
void Check(bool valid, const std::string& reason)
{
	if (!valid) throw std::runtime_error(reason);
}

void Compare(CSPrimitives& primitive, const std::function<bool(const double*)>& oracle={})
{
	Check(primitive.Update(),"fixture update failed");
	std::vector<double> data;
	GPU_SetupExpressionBank bank;
	const GPU_SetupPrimitive descriptor=GPU_DescribePrimitive(primitive,data,&bank);
	Check(descriptor.version==2 && descriptor.id==primitive.GetID(),"descriptor identity/version lost");
	for (int x=-7;x<=7;++x) for (int y=-7;y<=7;++y)
	{
		const double bounds[6]={double(x),double(x+1),double(y),double(y+2),-2,2};
		Check(GPU_SetupCandidate(descriptor,bounds)==(primitive.IsInsideBox(bounds)>=0),"candidate bound decision differs from CSXCAD");
	}
	std::vector<double> coordinates;
	for (int x=-16;x<=16;++x) for (int y=-16;y<=16;++y) for (int z=-8;z<=8;++z)
	{
		const double p[3]={x*0.25,y*0.25,z*0.25};
		coordinates.insert(coordinates.end(),p,p+3);
	}
	std::vector<unsigned char> results(coordinates.size()/3);
#ifdef OPENEMS_DEVICE_GEOMETRY_TEST
	GPU_TestGeometrySamples(descriptor,data,bank,coordinates,results);
#else
	std::vector<double> stack(std::max(1u,bank.stack_size));
	for (size_t i=0;i<results.size();++i)
	{
		int error=0;
		GPU_SetupExpressionView expressions={bank.programs.data(),bank.instructions.data(),bank.constants.data(),stack.data(),bank.stack_size,&error};
		results[i]=GPU_SetupContains(descriptor,data.data(),coordinates.data()+3*i,nullptr,&expressions);
		Check(error==0,"geometry expression failed");
	}
#endif
	for (size_t i=0;i<results.size();++i)
	{
		const double* p=coordinates.data()+3*i;
		const bool expected=primitive.IsInside(p);
		if ((oracle && expected!=oracle(p)) || bool(results[i])!=expected)
		{
			std::ostringstream message;
			message << primitive.GetTypeName() << " containment differs at " << p[0] << "," << p[1] << "," << p[2];
			throw std::runtime_error(message.str());
		}
	}
	std::cout << "PASS " << primitive.GetTypeName() << std::endl;
}

void AddCube(CSPrimPolyhedron& poly, float lo, float hi, bool reverse=false, float shift=0)
{
	const unsigned int offset=poly.GetNumVertices();
	const float points[][3]={{lo,lo,lo},{hi,lo,lo},{hi,hi,lo},{lo,hi,lo},
	                         {lo,lo,hi},{hi,lo,hi},{hi,hi,hi},{lo,hi,hi}};
	const int triangles[][3]={{0,2,1},{0,3,2},{4,5,6},{4,6,7},{0,1,5},{0,5,4},
	                          {1,2,6},{1,6,5},{2,3,7},{2,7,6},{3,0,4},{3,4,7}};
	for (const auto& p:points) poly.AddVertex(p[0]+shift,p[1],p[2]);
	for (const auto& t:triangles)
	{
		std::vector<int> face={int(offset)+t[0],int(offset)+t[1],int(offset)+t[2]};
		if (reverse) std::reverse(face.begin(),face.end());
		poly.AddFace(face);
	}
}

#ifdef OPENEMS_DEVICE_GEOMETRY_TEST
// A closed, shared-vertex surface with 49,152 triangles. Its independent oracle
// is the same cube, so subdivision cannot change its containment semantics.
void AddSubdividedCube(CSPrimPolyhedron& poly)
{
	const int divisions=64;
	std::map<std::array<int,3>,int> vertices;
	auto vertex=[&](const std::array<int,3>& p) {
		auto found=vertices.find(p);
		if (found!=vertices.end()) return found->second;
		const int index=static_cast<int>(poly.GetNumVertices());
		poly.AddVertex(-2.0f+4.0f*p[0]/divisions,-2.0f+4.0f*p[1]/divisions,-2.0f+4.0f*p[2]/divisions);
		vertices[p]=index;return index;
	};
	for (int axis=0;axis<3;++axis) for (int side=0;side<2;++side)
		for (int i=0;i<divisions;++i) for (int j=0;j<divisions;++j)
		{
			const int a=(axis+1)%3,b=(axis+2)%3;
			std::array<int,3> p={{0,0,0}};p[axis]=side*divisions;p[a]=i;p[b]=j;
			const int q0=vertex(p);++p[a];const int q1=vertex(p);
			++p[b];const int q2=vertex(p);--p[a];const int q3=vertex(p);
			poly.AddFace(side ? std::vector<int>{q0,q1,q2} : std::vector<int>{q2,q1,q0});
			poly.AddFace(side ? std::vector<int>{q0,q2,q3} : std::vector<int>{q3,q2,q0});
		}
}
#endif

bool CubeInside(const double* p, double lo, double hi, double shift=0)
{
	return p[0]>=lo+shift && p[0]<=hi+shift && p[1]>=lo && p[1]<=hi && p[2]>=lo && p[2]<=hi;
}

bool CubeSurface(const double* p, double lo, double hi)
{
	return CubeInside(p,lo,hi) && (p[0]==lo || p[0]==hi || p[1]==lo || p[1]==hi || p[2]==lo || p[2]==hi);
}

struct GeometryFile
{
	std::string path;
	explicit GeometryFile(const char* extension)
	{
		static unsigned int count=0;
		path="gpu-setup-geometry-"+std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count())+
		     "-"+std::to_string(++count)+extension;
	}
	~GeometryFile() {std::remove(path.c_str());}
};

void CompareImported(CSPrimPolyhedron& native, ParameterSet& parameters, CSProperties& material, bool ply)
{
	GeometryFile fixture(ply ? ".ply" : ".stl");
	std::ofstream file(fixture.path);
	Check(bool(file),"cannot create imported geometry fixture");
	if (ply)
	{
		file << "ply\nformat ascii 1.0\nelement vertex " << native.GetNumVertices()
		     << "\nproperty float x\nproperty float y\nproperty float z\nelement face " << native.GetNumFaces()
		     << "\nproperty list uchar int vertex_indices\nend_header\n";
		for (unsigned int i=0;i<native.GetNumVertices();++i)
		{
			const float* p=native.GetVertex(i);file << p[0] << " " << p[1] << " " << p[2] << "\n";
		}
		for (unsigned int i=0;i<native.GetNumFaces();++i)
		{
			unsigned int count=0;const int* face=native.GetFace(i,count);file << count;
			for (unsigned int k=0;k<count;++k) file << " " << face[k];file << "\n";
		}
	}
	else
	{
		file << "solid generated_openems_fixture\n";
		for (unsigned int i=0;i<native.GetNumFaces();++i)
		{
			unsigned int count=0;const int* face=native.GetFace(i,count);Check(count==3,"STL fixture requires triangles");
			file << "facet normal 0 0 0\nouter loop\n";
			for (unsigned int k=0;k<count;++k)
			{
				const float* p=native.GetVertex(face[k]);file << "vertex " << p[0] << " " << p[1] << " " << p[2] << "\n";
			}
			file << "endloop\nendfacet\n";
		}
		file << "endsolid generated_openems_fixture\n";
	}
	file.close();Check(bool(file),"imported fixture write failed");
	CSPrimPolyhedronReader* reader=new CSPrimPolyhedronReader(&parameters,&material);
	reader->SetFilename(fixture.path);reader->SetFileType(ply ? CSPrimPolyhedronReader::PLY_FILE : CSPrimPolyhedronReader::STL_FILE);
	Check(reader->ReadFile(),"generated geometry import failed");
	Compare(*reader,[](const double* p) {return CubeInside(p,-3,-1);});
}

void Polygon(CSPrimPolygon& polygon)
{
	polygon.SetNormDir(2); polygon.SetElevation(0.0);
	const double vertices[]={-2,-2,2,-2,2,0,0,0,0,2,-2,2};
	for (double value:vertices) polygon.AddCoord(value);
}

class CustomBox : public CSPrimBox
{
public:
	CustomBox(ParameterSet* parameters, CSProperties* property) : CSPrimBox(parameters,property) {}
};

void CheckPropertyIDs()
{
	ContinuousStructure structure;
	for (int i=0;i<4;++i)
	{
		auto* material=new CSPropMaterial(structure.GetParameterSet());structure.AddProperty(material);
		material->SetEpsilon(2.5); // Equal numeric values must not hide different IDs.
		if (i==1)
		{
			auto* sphere=new CSPrimSphere(structure.GetParameterSet(),material);sphere->SetRadius(1.75);sphere->SetPriority(2);
		}
		else
		{
			auto* box=new CSPrimBox(structure.GetParameterSet(),material);box->SetPriority(i==3 ? 5 : i==2 ? 2 : 0);
			for (int n=0;n<3;++n) {box->SetCoord(2*n,i==0 ? -3.0 : -1.0);box->SetCoord(2*n+1,i==0 ? 3.0 : 1.0);}
			if (i==3) {const double shift[3]={1.5,-0.5,0.25};box->GetTransform()->Translate(shift);}
		}
	}
	// More than a small insertion-sort partition exposes unstable priority ties.
	for (int i=0;i<32;++i)
	{
		auto* material=new CSPropMaterial(structure.GetParameterSet());structure.AddProperty(material);material->SetEpsilon(2.5);
		auto* box=new CSPrimBox(structure.GetParameterSet(),material);box->SetPriority(2);
		for (int n=0;n<3;++n) {box->SetCoord(2*n,-1.5);box->SetCoord(2*n+1,1.5);}
	}
	for (CoordinateSystem input:{CARTESIAN,CYLINDRICAL})
	{
		structure.SetCoordInputType(input);Check(structure.Update().empty(),"property ID fixture failed");
		std::vector<GPU_SetupPrimitive> shapes;std::vector<double> data,coordinates;GPU_SetupExpressionBank bank;
		for (auto* primitive:structure.GetAllPrimitives(true,CSProperties::MATERIAL)) shapes.push_back(GPU_DescribePrimitive(*primitive,data,&bank));
		for (int x=-12;x<=12;++x) for (int y=-12;y<=12;++y) for (int z=-5;z<=5;++z)
		{const double point[3]={x*0.25,y*0.25,z*0.5};coordinates.insert(coordinates.end(),point,point+3);}
		std::vector<uint32_t> actual(coordinates.size()/3);
#ifdef OPENEMS_DEVICE_GEOMETRY_TEST
		GPU_TestPropertyIDs(shapes,data,coordinates,actual);
#else
		for (size_t p=0;p<actual.size();++p) for (const auto& shape:shapes)
			if (GPU_SetupContains(shape,data.data(),coordinates.data()+3*p)) {actual[p]=shape.property_id+1;break;}
#endif
		for (size_t p=0;p<actual.size();++p)
		{
			auto candidates=structure.GetAllPrimitives(true,CSProperties::MATERIAL);
			auto* expected=structure.GetPropertyByCoordPriority(coordinates.data()+3*p,candidates);
			Check(actual[p]==(expected ? expected->GetUniqueID()+1 : 0),"exact material property ID/priority/tie snapshot differs");
		}
	}
	std::cout << "PASS exact material property IDs, overlapping priorities, equal-value ties and transforms" << std::endl;
}
}

int main()
{
	try
	{
		CheckPropertyIDs();
		ParameterSet parameters;
		CSPropMaterial material(&parameters);
		CSPrimBox* box=new CSPrimBox(&parameters,&material);
		for (int n=0;n<3;++n) { box->SetCoord(2*n,-1.0); box->SetCoord(2*n+1,1.0); }
		Compare(*box);
		box->SetCoordinateSystem(CYLINDRICAL);
		box->SetCoord(0,1.0); box->SetCoord(1,3.0); box->SetCoord(2,-0.5); box->SetCoord(3,4.5);
		Compare(*box);
		box->SetCoord(2,6*acos(-1.0)+0.5); box->SetCoord(3,6*acos(-1.0)+5.0);
		Compare(*box);
		box->SetCoordinateSystem(CARTESIAN);
		for (int n=0;n<3;++n) { box->SetCoord(2*n,-1.0); box->SetCoord(2*n+1,1.0); }
		const double matrix[]={1,0.25,0,0.5, 0,1,0,-0.25, 0,0,1,0, 0.1,0,0,1};
		box->GetTransform()->SetMatrix(matrix,false); Compare(*box);
		CSPrimSphere* sphere=new CSPrimSphere(&parameters,&material);
		sphere->SetCenter(-1.0,0.0,0.0); sphere->SetRadius(1.5); Compare(*sphere);
		CSPrimSphericalShell* spherical=new CSPrimSphericalShell(&parameters,&material);
		spherical->SetRadius(1.5); spherical->SetShellWidth(0.5); Compare(*spherical);
		CSPrimCylinder* cylinder=new CSPrimCylinder(&parameters,&material);
		cylinder->SetCoord(4,-1.0); cylinder->SetCoord(5,1.0); cylinder->SetRadius(1.0); Compare(*cylinder);
		CSPrimCylindricalShell* cylindrical=new CSPrimCylindricalShell(&parameters,&material);
		cylindrical->SetCoord(4,-1.0); cylindrical->SetCoord(5,1.0); cylindrical->SetRadius(1.0); cylindrical->SetShellWidth(0.5); Compare(*cylindrical);
		CSPrimPolygon* polygon=new CSPrimPolygon(&parameters,&material); Polygon(*polygon); Compare(*polygon);
		CSPrimLinPoly* linear=new CSPrimLinPoly(&parameters,&material); Polygon(*linear); linear->SetLength(-1.0); Compare(*linear);
		CSPrimRotPoly* rotation=new CSPrimRotPoly(&parameters,&material); Polygon(*rotation);
		rotation->SetRotAxisDir(0); rotation->SetAngle(0,5.0); rotation->SetAngle(1,2.0); Compare(*rotation);
		CSPrimMultiBox* boxes=new CSPrimMultiBox(&parameters,&material);
		const double coordinates[]={-2,-1,-1,1,-1,1, 2,1,2,1,1,-1, 0,0,-1,1,-1,1};
		for (double value:coordinates) boxes->AddCoord(value);
		Compare(*boxes);
		CSPrimWire* wire=new CSPrimWire(&parameters,&material);
		double a[3]={-2,0,0},b[3]={0,0,0},c[3]={0,0,0},d[3]={1,1,1};
		wire->AddPoint(a); wire->AddPoint(b); wire->AddPoint(c); wire->AddPoint(d); wire->SetWireRadius(0.5); Compare(*wire);
		Compare(*new CSPrimPoint(&parameters,&material));
		CSPrimCurve* curve=new CSPrimCurve(&parameters,&material);
		curve->AddPoint(a); curve->AddPoint(d); Compare(*curve);
		for (CSPrimitives* unsupported:std::vector<CSPrimitives*>{new CSPrimPolyhedron(&parameters,&material),
		     new CSPrimPolyhedronReader(&parameters,&material),new CSPrimUserDefined(&parameters,&material)})
		{
			std::vector<double> data;bool rejected=false;
			try {GPU_DescribePrimitive(*unsupported,data);} catch (const GPU_SetupUnavailable&) {rejected=true;}
			Check(rejected,"unsupported geometry was accepted for GPU construction");
		}
		for (CSPrimitives* primitive:std::vector<CSPrimitives*>{box,sphere,spherical,cylinder,cylindrical,polygon,linear,rotation,boxes,wire,curve})
		{
			primitive->SetCoordInputType(CYLINDRICAL);Compare(*primitive);
		}
		CustomBox* custom=new CustomBox(&parameters,&material);
		bool rejected=false;
		try { std::vector<double> data; GPU_DescribePrimitive(*custom,data); } catch (const GPU_SetupUnavailable&) { rejected=true; }
		Check(rejected,"custom primitive inherited device compatibility");
		bool supported=false;
		GPU_RegisterPrimitiveAdapter(typeid(CustomBox),"test Box adapter",
			[&supported](CSPrimitives&,std::string& reason) {reason="test adapter is disabled";return supported;},
			[](CSPrimitives& primitive,std::vector<double>& data,GPU_SetupExpressionBank* bank) {
				return GPU_DescribePrimitiveAs(primitive,typeid(CSPrimBox),data,bank);
			});
		struct RemoveTestAdapter {~RemoveTestAdapter() {GPU_UnregisterPrimitiveAdapter(typeid(CustomBox));}} remove_adapter;
		rejected=false;
		try {std::vector<double> data;GPU_DescribePrimitive(*custom,data);} catch (const GPU_SetupUnavailable& error) {
			rejected=std::string(error.what()).find("disabled")!=std::string::npos;
			if (!rejected) std::cerr << error.what() << "; actual=" << typeid(*custom).name() << "; registered=" << typeid(CustomBox).name() << std::endl;
		}
		Check(rejected,"registered adapter preparation check was skipped");
		supported=true;
		for (int n=0;n<3;++n) {custom->SetCoord(2*n,-1);custom->SetCoord(2*n+1,1);}
		Compare(*custom);
		const double singular[16]={0}; box->GetTransform()->SetMatrix(singular,false); rejected=false;
		try { std::vector<double> data; GPU_DescribePrimitive(*box,data); } catch (const GPU_SetupUnavailable&) { rejected=true; }
		Check(rejected,"singular transform accepted");
		rejected=false;
		try { GPU_SetupMultiply(std::numeric_limits<size_t>::max(),2); } catch (const GPU_SetupUnavailable&) { rejected=true; }
		Check(rejected,"construction size multiplication overflow accepted");
		rejected=false;
		try { GPU_SetupAdd(std::numeric_limits<size_t>::max(),1); } catch (const GPU_SetupUnavailable&) { rejected=true; }
		Check(rejected,"construction size addition overflow accepted");
		Check(GPU_SetupConstant("1/2") && GPU_SetupConstant("sin(pi/2)") && GPU_SetupConstant("if(0,1/0,1)"),"constant expression rejected");
		Check(!GPU_SetupConstant("1+x/2"),"spatial expression treated as constant");
		rejected=false;
		try {GPU_SetupConstant("1/0");} catch (const std::invalid_argument&) {rejected=true;}
		Check(rejected,"invalid constant expression accepted");
		return 0;
	}
	catch (const std::exception& error) { std::cerr << error.what() << std::endl; return 1; }
}
