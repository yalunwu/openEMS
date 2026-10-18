/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#include "openems_global.h"
extern "C" OPENEMS_EXPORT int openEMS_TestGPUHostFields();

#ifndef OPENEMS_NATIVE_SETUP_TESTS
int main() {return openEMS_TestGPUHostFields();}
#else
#include "FDTD/engine_gpu.h"
#include "FDTD/gpu_backend.h"
#include "ContinuousStructure.h"
#include <iostream>
#include <memory>
#include <algorithm>
#include <atomic>
#include <thread>

namespace
{
class FieldBackend : public GPU_Backend
{
public:
	std::vector<float> fields[2];
	unsigned int downloads=0,ranges=0,uploads=0;
	std::string GetName() const override {return "field-storage-test";}
	bool Init(const Operator* op) override
	{
		size_t size=3;
		for (int n=0;n<3;++n) size*=op->GetNumberOfLines(n,true);
		fields[0].assign(size,0);fields[1].assign(size,0);return true;
	}
	void UpdateVoltages() override {for (size_t i=0;i<fields[0].size();++i) fields[0][i]=float(i)*0.25f;}
	void UpdateCurrents() override {for (size_t i=0;i<fields[1].size();++i) fields[1][i]=-float(i)*0.5f;}
	void DownloadVoltages(ArrayLib::ArrayNIJK<float>& field) override {++downloads;std::copy(fields[0].begin(),fields[0].end(),field.data());}
	void DownloadCurrents(ArrayLib::ArrayNIJK<float>& field) override {++downloads;std::copy(fields[1].begin(),fields[1].end(),field.data());}
	void UploadVoltages(const ArrayLib::ArrayNIJK<float>& field) override {++uploads;std::copy(field.data(),field.data()+field.size(),fields[0].begin());}
	void UploadCurrents(const ArrayLib::ArrayNIJK<float>& field) override {++uploads;std::copy(field.data(),field.data()+field.size(),fields[1].begin());}
	bool DownloadRange(bool current,size_t offset,size_t count,float* values) override
	{
		++ranges;std::copy_n(fields[current].data()+offset,count,values);return true;
	}
	GPU_Backend* NewSubGridBackend() override {return new FieldBackend;}
	GPU_MultiGridLink* CreateMultiGridLink(GPU_Backend*,const GPU_MultiGridInterpolation&) override {return nullptr;}
	GPU_Extension* CreateExtension(Engine_Extension*,Engine*) override {return nullptr;}
};

void CheckFields(bool condition) {if (!condition) throw std::runtime_error("lazy GPU host field contract failed");}
}

int openEMS_TestGPUHostFields()
{
	try
	{
		ContinuousStructure structure;
		for (int n=0;n<3;++n) for (int p=0;p<5;++p) structure.GetGrid()->AddDiscLine(n,p);
		std::unique_ptr<Operator> op(Operator::New());
		CheckFields(op->SetGeometryCSX(&structure));
		FieldBackend* backend=new FieldBackend;
		std::unique_ptr<Engine_GPU> engine(Engine_GPU::New(op.get(),"field-storage-test",nullptr,backend));
		CheckFields(!engine->FieldsOnHost() && engine->HostFieldStorageBytes()==0);
		CheckFields(engine->GetVolt(0,0,0,0)==0 && engine->GetCurr(0,0,0,0)==0 && backend->downloads==0);
		engine->IterateTS(1);
		CheckFields(engine->GetVolt(0,0,0,1)==0.25f && engine->GetCurr(0,0,0,1)==-0.5f);
		CheckFields(engine->HostFieldStorageBytes()==0 && backend->downloads==0 && backend->ranges==2);
		const auto& voltages=engine->HostVoltages();
		CheckFields(voltages(0,0,0,1)==0.25f && engine->HostFieldStorageBytes()==3*125*sizeof(float) && backend->downloads==1);
		engine->SetVolt(0,0,0,1,123.0f);
		CheckFields(backend->fields[0][1]==123.0f && backend->uploads==1);
		engine->ForceHostFallback();
		CheckFields(engine->FieldsOnHost() && engine->HostFieldStorageBytes()==6*125*sizeof(float));
		CheckFields(engine->GetCurr(0,0,0,1)==-0.5f && engine->GetVolt(0,0,0,1)==123.0f);
		std::cout << "PASS device initialization, lazy line reads, host materialization and fallback fields" << std::endl;
		struct ConcurrentTable : GPU_FieldGather
		{
			mutable std::atomic<unsigned int> reads{0};
			size_t Count() const override {return 4096;}
			void Read(std::vector<Engine_GatherEntry>& output) const override
			{++reads;std::this_thread::yield();output.resize(Count());for (auto& entry:output) {entry={};entry.form=Engine_GatherEntry::RAW;}}
		};
		auto table=std::make_shared<ConcurrentTable>();Prepared_Field_Gather_FDTD gather(nullptr,nullptr,false,1,1);gather.prepared=table;
		std::vector<std::thread> readers;
		for (int n=0;n<16;++n) readers.emplace_back([&gather]() {gather.EnsureHost();});
		for (auto& reader:readers) reader.join();
		CheckFields(table->reads==1 && gather.entries.size()==table->Count());
		std::cout << "PASS concurrent sampling consumers materialize one complete table" << std::endl;
		return 0;
	}
	catch (const std::exception& error) {std::cerr << error.what() << std::endl;return 1;}
}
#endif
