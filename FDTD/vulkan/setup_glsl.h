/*
 * Copyright (C) 2026 openEMS Project
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef VULKAN_SETUP_GLSL_H
#define VULKAN_SETUP_GLSL_H

// Constants for dimensions/grid offsets are prepended by the construction code.
static const char* const kVulkanSetupShaderParts[] = { R"glsl(
layout(local_size_x=256) in;
layout(std430,binding=0) readonly buffer Grid { double grid[]; };
layout(std430,binding=1) readonly buffer Geometry { double geo[]; };
layout(std430,binding=2) readonly buffer Candidates { uint cand[]; };
layout(std430,binding=3) buffer Used { uint used[]; };
layout(std430,binding=4) buffer EC0 { float ec0[]; };
layout(std430,binding=5) buffer EC1 { float ec1[]; };
layout(std430,binding=6) buffer EC2 { float ec2[]; };
layout(std430,binding=7) buffer EC3 { float ec3[]; };
layout(std430,binding=8) buffer VV { float vv[]; };
layout(std430,binding=9) buffer VI { float vi[]; };
layout(std430,binding=10) buffer II { float ii[]; };
layout(std430,binding=11) buffer IV { float iv[]; };
layout(std430,binding=12) buffer Material0 { float material0[]; };
layout(std430,binding=13) buffer Material1 { float material1[]; };
layout(std430,binding=14) buffer Material2 { float material2[]; };
layout(std430,binding=15) buffer Material3 { float material3[]; };
layout(std430,binding=16) buffer Reduction { double reduction[]; };
layout(std430,binding=17) readonly buffer Auxiliary { double aux[]; };
layout(std430,binding=18) buffer ExtensionCoefficients { float ext[]; };
layout(std430,binding=19) buffer UpmlIndices { uint upmlIndices[]; };
layout(std430,binding=20) buffer UpmlVoltage { vec4 upmlVoltage[]; };
layout(std430,binding=21) buffer UpmlCurrent { vec4 upmlCurrent[]; };
layout(push_constant) uniform Params {
    uint mode; uint base; double dt; uint average; uint method; uint stores; uint reserved;
} pc;
shared double minima[256];
shared uint winners[256];
uint linear(ivec3 p) { return uint(p.x)*DIMS.y*DIMS.z+uint(p.y)*DIMS.z+uint(p.z); }
ivec3 position(uint id) { return ivec3(id/(DIMS.y*DIMS.z),(id/DIMS.z)%DIMS.y,id%DIMS.z); }
double g(int n,int p,int kind) { return grid[OFFSETS[n]+uint(p)*6+uint(kind)]; }
double width(int n,int p,bool dual) { return g(n,p,dual?5:4); }
double area(int n,ivec3 p,bool dual) {
    if(any(lessThan(p,ivec3(0)))) return 0.0;
    return width((n+1)%3,p[(n+1)%3],!dual)*width((n+2)%3,p[(n+2)%3],!dual);
}
bool inside(uint prim,dvec3 coord) {
    uint b=prim*48;
    precise dvec3 p;
    for(int n=0;n<3;++n) {
        p[n]=0.0;
        for(int k=0;k<3;++k) p[n]+=geo[b+4+uint(n*4+k)]*coord[k];
        p[n]+=geo[b+4+uint(n*4+3)];
    }
    dvec3 a=dvec3(geo[b+16],geo[b+17],geo[b+18]);
    dvec3 z=dvec3(geo[b+19],geo[b+20],geo[b+21]);
    if(geo[b]==1.0) return all(greaterThanEqual(p,min(a,z)))&&all(lessThanEqual(p,max(a,z)));
    precise double distance=0.0;
    if(geo[b]==2.0) {
        for(int n=0;n<3;++n) distance+=(p[n]-a[n])*(p[n]-a[n]);
        return sqrt(distance)<geo[b+22];
    }
    for(int n=0;n<3;++n) if(p[n]<geo[b+23+uint(n*2)]||p[n]>geo[b+24+uint(n*2)]) return false;
    precise dvec3 dir=z-a;
    precise double ll=0.0, projection=0.0;
    for(int n=0;n<3;++n) { ll+=dir[n]*dir[n]; projection+=(p[n]-a[n])*dir[n]; }
    precise double foot=projection/ll;
    if(foot<0.0||foot>1.0) return false;
    for(int n=0;n<3;++n) { precise double d=p[n]-(a[n]+foot*dir[n]); distance+=d*d; }
    return sqrt(distance)<=geo[b+22];
}
uint property(dvec3 coord,ivec3 pos,uint which) {
    uint xy=uint(pos.x)*DIMS.y+uint(pos.y);
    uint row=which*(DIMS.x*DIMS.y+1)+xy;
    for(uint i=cand[row];i<cand[row+1];++i) {
        uint prim=cand[i];
        if(inside(prim,coord)) { atomicOr(used[prim],1); return prim; }
    }
    return 0;
}
uint sourceProperty(dvec3 coord,ivec3 pos) {
    uint row=2*(DIMS.x*DIMS.y+1)+uint(pos.z)*DIMS.y+uint(pos.y);
    for(uint i=cand[row];i<cand[row+1];++i) {
        uint prim=cand[i];
        if(inside(prim,coord)) { atomicOr(used[prim],1); return prim; }
    }
    return 0;
}
dvec4 material(int n,dvec3 coord,ivec3 pos) {
    uint b=property(coord,pos,0)*48+32+uint(n)*4;
    return dvec4(geo[b],geo[b+1],geo[b+2],geo[b+3]);
}
dvec4 effective(int n,ivec3 p) {
    int a=(n+1)%3,b=(n+2)%3;
    precise dvec4 m=dvec4(0.0);
    precise double A=0.0, length=0.0;
    dvec3 coord=dvec3(g(0,p.x,0),g(1,p.y,0),g(2,p.z,0));
    for(int q=0;q<4;++q) {
        ivec3 loc=p; loc[a]-=q%2; loc[b]-=q/2;
        dvec3 c;
        if(pc.average==0) {
            c=coord; c[n]+=g(n,p[n],2)*0.5;
            c[a]+=(q%2==0?g(a,p[a],2):-g(a,p[a],3))*0.25;
            c[b]+=(q/2==0?g(b,p[b],2):-g(b,p[b],3))*0.25;
        } else {
            if(any(lessThan(loc,ivec3(0)))) continue;
            for(int k=0;k<3;++k) c[k]=g(k,loc[k],1);
        }
        precise double weight=area(n,loc,true);
        precise dvec4 raw=material(n,c,p);
        m.x+=raw.x*weight; m.y+=raw.y*weight; A+=weight;
    }
    m.x*=8.85418781762e-12lf/A; m.y/=A;
    for(int q=0;q<2;++q) {
        ivec3 loc=p; loc[n]-=1-q;
        dvec3 c;
        if(pc.average==0) {
            c=coord; c[n]+=(q==0?-g(n,p[n],3):g(n,p[n],2))*0.25;
            c[a]+=g(a,p[a],2)*0.5; c[b]+=g(b,p[b],2)*0.5;
        } else {
            if(any(lessThan(loc,ivec3(0)))) continue;
            for(int k=0;k<3;++k) c[k]=g(k,loc[k],1);
        }
        precise double weight=any(lessThan(loc,ivec3(0)))?0.0:width(n,loc[n],false);
        precise dvec4 raw=material(n,c,p);
        m.z+=weight/raw.z;
        if(raw.w!=0.0) m.w+=weight/raw.w; else m.w=0.0;
        length+=weight;
    }
    m.z=length*1.256637062e-6lf/m.z;
    if(m.w!=0.0) m.w=length/m.w;
    return m;
}
uint shifted(ivec3 p) { return linear(clamp(p,ivec3(0),ivec3(DIMS)-1)); }
float reciprocal(int l,ivec3 lp,int c,ivec3 cp) {
    precise float product=ec2[uint(l)*CELLS+shifted(lp)]*ec0[uint(c)*CELLS+shifted(cp)];
    // Vulkan FP32 division may use an approximate reciprocal. Round a double
    // quotient once to reproduce the CPU float expression's rounding point.
    precise float value=float(1.0lf/double(product));
    return value;
}
double dtValue(int n,ivec3 p) {
    int a=(n+1)%3,b=(n+2)%3;
    ivec3 ea=ivec3(0),eb=ivec3(0),en=ivec3(0); ea[a]=1; eb[b]=1; en[n]=1;
    if(pc.method!=3) {
        precise float sum=float(4.0lf/double(ec2[uint(a)*CELLS+linear(p)]));
        sum+=float(4.0lf/double(ec2[uint(a)*CELLS+shifted(p-eb)]));
        sum+=float(4.0lf/double(ec2[uint(b)*CELLS+linear(p)]));
        sum+=float(4.0lf/double(ec2[uint(b)*CELLS+shifted(p-ea)]));
        precise float quotient=float(double(sum)/double(ec0[uint(n)*CELLS+linear(p)]));
        precise float root=float(sqrt(double(quotient)));
        precise float value=float(2.0lf/double(root));
        return double(value);
    }
    // Each reciprocal/product and pair sum is float in the CPU reference.
    precise float pair=reciprocal(b,p,n,p+ea)+reciprocal(b,p,n,p);
    precise double w=double(pair);
    pair=reciprocal(a,p,n,p+eb)+reciprocal(a,p,n,p); w+=double(pair);
    pair=reciprocal(b,p-ea,n,p)+reciprocal(b,p-ea,n,p-ea); w+=double(pair);
    pair=reciprocal(a,p-ea-eb,n,p-ea)+reciprocal(a,p-ea-eb,n,p-ea-eb); w+=double(pair);
    for(int q=0;q<2;++q) {
        ivec3 cp=p+q*en;
        precise dvec4 t=dvec4(double(reciprocal(b,p,a,cp)),double(reciprocal(b,p-ea,a,cp)),
                             double(reciprocal(a,p,b,cp)),double(reciprocal(a,p-eb,b,cp)));
        precise double total=t.x+t.y+t.z+t.w-2.0*min(min(t.x,t.y),min(t.z,t.w));
        w+=total;
    }
    return 2.0/sqrt(w);
}
)glsl", R"glsl(
// The host exports evaluated constants; geometry selection and ADE arithmetic
// are performed here in the same Yee locations and order as the reference.
void lorentz(int n,ivec3 p,uint pole,out dvec3 v,out dvec3 i) {
    v=dvec3(1.0,0.0,0.0); i=v;
    uint cell=uint(n)*CELLS+linear(p);
    for(int magnetic=0;magnetic<2;++magnetic) {
        int a=(n+1)%3,b=(n+2)%3;
        bool valid=magnetic==0?p[n]<int(DIMS[n])-1:
            p[a]<int(DIMS[a])-1&&p[b]<int(DIMS[b])-1;
        double coefficient=double(magnetic==0?vi[cell]:iv[cell]);
        if(!valid||coefficient==0.0) continue;
        dvec3 c;
        for(int axis=0;axis<3;++axis) c[axis]=g(axis,p[axis],(axis==n)!=(magnetic!=0)?1:0);
        uint prim=property(c,p,1);
        uint offset=4+(prim*uint(aux[1])+pole)*18+uint(n)*6+uint(magnetic)*3;
        precise double plasma=aux[offset]*2.0*3.1415926535897932384626433832795lf;
        precise double capacitance=double(magnetic==0?ec0[cell]:ec2[cell]);
        if(plasma<=0.0||capacitance<=0.0) continue;
        precise double reactance=1.0/(plasma*plasma*capacitance);
        precise double loss=aux[offset+1]>0.0?reactance/aux[offset+1]:0.0;
        precise double omega=aux[offset+2]*2.0*3.1415926535897932384626433832795lf;
        precise double other=omega>0.0?1.0/(omega*omega*reactance):0.0;
        precise dvec3 result=dvec3((2.0*reactance-pc.dt*loss)/(2.0*reactance+pc.dt*loss),
            pc.dt/(reactance+pc.dt*loss/2.0)*coefficient,other>0.0?pc.dt/other/coefficient:0.0);
        if(magnetic==0) v=result; else i=result;
    }
}
void debye(int n,ivec3 p,uint pole,out dvec2 result) {
    result=dvec2(0.0);
    uint cell=uint(n)*CELLS+linear(p);
    if(p[n]>=int(DIMS[n])-1||vi[cell]==0.0) return;
    dvec3 c=dvec3(g(0,p.x,0),g(1,p.y,0),g(2,p.z,0)); c[n]=g(n,p[n],1);
    uint prim=property(c,p,1),offset=4+(prim*uint(aux[1])+pole)*6+uint(n)*2;
    precise double delta=aux[offset],tau=aux[offset+1];
    if(delta<=0.0||tau<=0.0) return;
    if(tau<3.0*pc.dt) { atomicOr(used[ERROR_INDEX],2); return; }
    precise double C=8.85418781762e-12lf*delta*(area(n,p,false)/width(n,p[n],false));
    result=dvec2((2.0*tau-pc.dt)/(2.0*tau+pc.dt),C*double(vi[cell])/(2.0*tau+pc.dt));
}
int sheetTangent(uint prim,int n) {
    if(prim==0) return -1;
    uint offset=prim*48; int a=(n+1)%3,b=(n+2)%3,result=-1;
    if(geo[offset+23+uint(a)*2]!=geo[offset+24+uint(a)*2]) result=a;
    if(geo[offset+23+uint(b)*2]!=geo[offset+24+uint(b)*2]) result=b;
    return result;
}
dvec3 dispersiveCoefficients(int n,ivec3 p) {
    if(aux[0]==2.0) {
        uint mapped=floatBitsToUint(material2[linear(p)]);
        if(mapped==0) return dvec3(1.0,0.0,0.0);
        uint count=uint(aux[1]),offset=uint(aux[2])*6*count+uint(n)*count+mapped-1;
        return dvec3(double(material3[offset]),double(material3[offset+3*count]),0.0);
    }
    dvec3 v,i; lorentz(n,p,uint(aux[2]),v,i);
    return aux[3]==0.0?v:i;
}
void main() {
    uint id=pc.base+gl_GlobalInvocationID.x;
    if(pc.mode==23) {
        if(id<CELLS) material2[id]=uintBitsToFloat(floatBitsToUint(material0[id])!=0?
            uint(reduction[id/256])+floatBitsToUint(material1[id])+1:0);
        return;
    }
    if(pc.mode==21) {
        uint lane=gl_LocalInvocationID.x,selected=0;
        if(id<3*CELLS) {
            ivec3 p=position(id/3); int n=int(id%3);
            vec3 coeff=vec3(dispersiveCoefficients(n,p));
            selected=(coeff.x!=1.0||coeff.y!=0.0||coeff.z!=0.0)?1:0;
            material0[id]=uintBitsToFloat(selected);
        }
        winners[lane]=selected; barrier();
        for(uint step=1;step<256;step*=2) {
            uint value=lane>=step?winners[lane-step]:0;
            barrier(); winners[lane]+=value; barrier();
        }
        if(id<3*CELLS) material1[id]=uintBitsToFloat(winners[lane]-selected);
        if(lane==255) reduction[id/256]=double(winners[lane]);
        return;
    }
    if(pc.mode==22) {
        if(id>=3*CELLS||floatBitsToUint(material0[id])==0) return;
        uint slot=uint(reduction[id/256])+floatBitsToUint(material1[id]);
        ivec3 p=position(id/3); int n=int(id%3);
        dvec3 coeff=dispersiveCoefficients(n,p);
        ext[slot*4]=uintBitsToFloat(uint(n)*CELLS+linear(p));
        ext[slot*4+1]=float(coeff.x); ext[slot*4+2]=float(coeff.y); ext[slot*4+3]=float(coeff.z);
        return;
    }
    if(pc.mode==24||pc.mode==25) {
        if(id>=3*CELLS) return;
        uint node=id/3; int n=int(id%3);
        if(floatBitsToUint(material0[node])==0) return;
        uint slot=(uint(reduction[node/256])+floatBitsToUint(material1[node]))*3+uint(n),poles=uint(aux[1]);
        ivec3 p=position(node);
        precise double sum=0.0;
        for(uint pole=0;pole<poles;++pole) {
            dvec2 coeff; debye(n,p,pole,coeff); sum+=coeff.y;
            if(pc.mode==25) { ext[(slot*poles+pole)*2]=float(coeff.x); ext[(slot*poles+pole)*2+1]=float(coeff.y); }
        }
        if(pc.mode==24) {
            ext[slot*4]=uintBitsToFloat(uint(n)*CELLS+node); ext[slot*4+1]=uintBitsToFloat(uint(aux[3])+slot*poles);
            ext[slot*4+2]=uintBitsToFloat(poles); ext[slot*4+3]=float(1.0/(1.0+sum));
        }
        return;
    }
    if(pc.mode==26) {
        if(id>=pc.reserved) return;
        uint src=id*10,dst=id*12;
        ext[dst]=uintBitsToFloat(floatBitsToUint(material3[src])+uint(aux[6])*CELLS);
        ext[dst+1]=material3[src+1]*material3[src+2]; ext[dst+2]=material3[src+6];
        ext[dst+3]=material3[src+3]; ext[dst+4]=material3[src+4]; ext[dst+5]=material3[src+5];
        ext[dst+6]=material3[src+7]; ext[dst+7]=material3[src+8]*material3[src+7]; ext[dst+8]=material3[src+9]*material3[src+7];
        ext[dst+9]=1.0; ext[dst+10]=0.0; ext[dst+11]=0.0;
        return;
    }
    if(pc.mode==27) {
        if(id>=pc.reserved) return;
        uint field=floatBitsToUint(ext[id*12]); bool last=true;
        for(uint q=id+1;q<pc.reserved;++q) if(floatBitsToUint(ext[q*12])==field) { last=false; break; }
        ext[id*12+9]=last?1.0:0.0;
        return;
    }
    if(pc.mode==30) {
        uint count=pc.reserved; if(id>=2*count) return;
        uint node=id/2; int axis=int(aux[0]),a=(axis+1)%3,b=(axis+2)%3,n=id%2==0?a:b;
        ivec3 p=ivec3(0); p[axis]=int(aux[1]); p[a]=int(node/DIMS[b]); p[b]=int(node%DIMS[b]);
        ext[id*4]=uintBitsToFloat(uint(n)*CELLS+linear(p)); p[axis]+=p[axis]==0?1:-1;
        ext[id*4+1]=uintBitsToFloat(uint(n)*CELLS+linear(p));
        ext[id*4+2]=material3[(id%2)*count+node]; ext[id*4+3]=uintBitsToFloat(0);
        return;
    }
    if(pc.mode==31) {
        if(id<pc.reserved) ext[(uint(aux[0])+id)*4+3]=uintBitsToFloat(uint(aux[1]));
        return;
    }
    if(pc.mode==28) {
        uint lane=gl_LocalInvocationID.x,selected=0,count=pc.reserved;
        if(id<2*count) selected=material3[((aux[22]!=0.0?2:0)*count+id)*3+2]!=0.0?1:0;
        if(id<3*CELLS) material0[id]=uintBitsToFloat(selected);
        winners[lane]=selected; barrier();
        for(uint step=1;step<256;step*=2) {
            uint value=lane>=step?winners[lane-step]:0;
            barrier(); winners[lane]+=value; barrier();
        }
        if(id<3*CELLS) material1[id]=uintBitsToFloat(winners[lane]-selected);
        if(lane==255) reduction[id/256]=double(winners[lane]);
        return;
    }
    if(pc.mode==29) {
        uint count=pc.reserved;
        if(id>=2*count||floatBitsToUint(material0[id])==0) return;
        uint slot=uint(reduction[id/256])+floatBitsToUint(material1[id]),node=id%count,component=id/count;
        bool voltage=aux[22]!=0.0; int axis=int(aux[0]),side=int(aux[1]),a=(axis+1)%3,b=(axis+2)%3,n=component==0?a:b;
        ivec3 start=ivec3(aux[2],aux[3],aux[4]),stop=ivec3(aux[5],aux[6],aux[7]),p=start;
        p[a]+=int(node/uint(stop[b]-start[b]+1)); p[b]+=int(node%uint(stop[b]-start[b]+1));
        p[axis]=side==0?start[axis]-(voltage?0:1):stop[axis];
        uint raw=((voltage?2:0)*count+id)*3;
        precise float amplitude=material3[raw+2],delta=material3[raw+1];
        ext[slot*4]=uintBitsToFloat(uint(n)*CELLS+linear(p)); ext[slot*4+1]=material3[raw];
        ext[slot*4+2]=(1.0-delta)*amplitude; ext[slot*4+3]=delta*amplitude;
        return;
    }
    if(pc.mode==32||pc.mode==33) {
        uint count=pc.reserved; if(id>=2*count) return;
        uint node=id/2,component=id%2; int axis=int(aux[2]),a=(axis+1)%3,b=(axis+2)%3,n=component==0?a:b;
        bool voltage=pc.mode==32,positive=aux[8]!=0.0;
        uint widthB=uint(aux[7])-(voltage?0:1),row=node/widthB,column=node%widthB;
        ivec3 p=ivec3(aux[3],aux[4],aux[5]); p[a]+=int(row); p[b]+=int(column);
        if(!voltage&&!positive) --p[axis];
        ext[id*4]=uintBitsToFloat(uint(n)*CELLS+linear(p)); p[axis]+=positive?1:-1;
        ext[id*4+1]=uintBitsToFloat(uint(n)*CELLS+linear(p));
        uint originalCount=uint(aux[6])*uint(aux[7]),original=row*uint(aux[7])+column;
        ext[id*4+2]=material3[component*originalCount+original];
        ext[id*4+3]=voltage?0.0:material3[(component+2)*originalCount+original];
        return;
    }
    if(pc.mode==19) {
        uint lane=gl_LocalInvocationID.x,selectedCell=0;
        if(id<CELLS) {
            ivec3 p=position(id); bool disabled=false;
            for(int axis=0;axis<3;++axis) if(uint(p[axis])<=BC_SIZE[2*axis]||uint(p[axis])>=DIMS[axis]-BC_SIZE[2*axis+1]-1) disabled=true;
            for(int n=0;n<3;++n) {
                uint cell=uint(n)*CELLS+id,prim=0;
                if(p[n]<int(DIMS[n])-1) {
                    dvec3 c=dvec3(g(0,p.x,0),g(1,p.y,0),g(2,p.z,0)); c[n]=g(n,p[n],1);
                    uint selected=property(c,p,1),offset=selected*48;
                    if(geo[offset+29]!=0.0) {
                        if(geo[offset+31]!=2.0||disabled||float(geo[offset+32])<=0.0||float(geo[offset+33])<=0.0) {
                            vv[cell]=0.0; vi[cell]=0.0; atomicAdd(used[PEC_INDEX+uint(n)],1);
                        } else { prim=selected; selectedCell=1; }
                    }
                }
                material2[cell]=uintBitsToFloat(prim);
            }
            material0[id]=uintBitsToFloat(selectedCell);
        }
        winners[lane]=selectedCell; barrier();
        for(uint step=1;step<256;step*=2) {
            uint value=lane>=step?winners[lane-step]:0;
            barrier(); winners[lane]+=value; barrier();
        }
        if(id<CELLS) material1[id]=uintBitsToFloat(winners[lane]-selectedCell);
        if(lane==255) reduction[id/256]=double(winners[lane]);
        return;
    }
    if(pc.mode==20) {
        if(id>=3*CELLS) return;
        uint node=id%CELLS; int n=int(id/CELLS);
        if(floatBitsToUint(material0[node])==0) return;
        uint slot=uint(reduction[node/256])+floatBitsToUint(material1[node]);
        uint outIndex=uint(n)*pc.reserved+slot,total=3*pc.reserved;
        for(uint kind=0;kind<4;++kind) ext[kind*total+outIndex]=0.0;
        uint prim=floatBitsToUint(material2[id]); int tangent=sheetTangent(prim,n);
        if(tangent<0) return;
        ivec3 p=position(node),previous=p; --previous[tangent];
        precise float conductivity=float(geo[prim*48+32]),thickness=float(geo[prim*48+33]);
        precise float G0=conductivity*thickness;
        precise float w0=float(8.0/double(G0)/double(thickness)/1.256637062e-6lf);
        precise float wstop=float(aux[0]*2.0*3.1415926535897932384626433832795lf),omega=wstop/w0;
        uint table=0; while(table<29&&aux[2+table*6]<=double(omega)) ++table;
        precise float factor=(sheetTangent(floatBitsToUint(material2[uint(tangent)*CELLS+node]),tangent)<0||
            sheetTangent(floatBitsToUint(material2[uint(tangent)*CELLS+linear(previous)]),tangent)<0)?2.0:1.0;
        precise float wtl=float(width(n,p[n],false)/width(tangent,p[tangent],true));
        precise float L1=float(aux[4+table*6]/double(G0)/double(w0)*double(factor));
        precise float L2=float(aux[5+table*6]/double(G0)/double(w0)*double(factor));
        precise float R1=float(aux[6+table*6]/double(G0)*double(factor));
        precise float R2=float(aux[7+table*6]/double(G0)*double(factor));
        precise float conductance=float(double(G0)*aux[3+table*6]/double(factor));
        L1*=wtl; L2*=wtl; R1*=wtl; R2*=wtl; conductance/=wtl;
        ec1[id]=conductance;
        ec0[id]=float(pc.dt*pc.dt/4.0*(16.0/double(min(L1,L2))+double(1.0/L1)+double(1.0/L2)));
        precise double C=double(ec0[id]),G=double(ec1[id]);
        vv[id]=float((1.0-pc.dt*G/2.0/C)/(1.0+pc.dt*G/2.0/C));
        vi[id]=float((pc.dt/C)/(1.0+pc.dt*G/2.0/C));
        precise double inductance=double(ec2[id]),resistance=double(ec3[id]);
        ii[id]=inductance>0.0?float((1.0-pc.dt*resistance/2.0/inductance)/(1.0+pc.dt*resistance/2.0/inductance)):0.0;
        iv[id]=inductance>0.0?float((pc.dt/inductance)/(1.0+pc.dt*resistance/2.0/inductance)):0.0;
        ext[outIndex]=float((2.0*double(L1)-pc.dt*double(R1))/(2.0*double(L1)+pc.dt*double(R1)));
        ext[total+outIndex]=float(pc.dt/(double(L1)+pc.dt*double(R1)/2.0)*double(vi[id]));
        ext[2*total+outIndex]=float((2.0*double(L2)-pc.dt*double(R2))/(2.0*double(L2)+pc.dt*double(R2)));
        ext[3*total+outIndex]=float(pc.dt/(double(L2)+pc.dt*double(R2)/2.0)*double(vi[id]));
        return;
    }
)glsl", R"glsl(
    if(pc.mode==12) {
        uint lane=gl_LocalInvocationID.x,flags=0;
        if(id<CELLS) {
            ivec3 p=position(id);
            for(int n=0;n<3;++n) {
                if(aux[0]==0.0) {
                    dvec3 v,i; lorentz(n,p,uint(aux[2]),v,i);
                    if(v.y!=0.0) flags|=1; if(i.y!=0.0) flags|=2;
                    if(v.z!=0.0) flags|=4; if(i.z!=0.0) flags|=8;
                } else for(uint pole=0;pole<uint(aux[1]);++pole) {
                    dvec2 coeff; debye(n,p,pole,coeff); if(coeff.y!=0.0) flags|=1;
                }
            }
            material0[id]=uintBitsToFloat(flags);
            if(aux[0]==0.0) atomicOr(used[ERROR_INDEX],flags<<4);
        }
        winners[lane]=flags!=0?1:0; barrier();
        for(uint step=1;step<256;step*=2) {
            uint value=lane>=step?winners[lane-step]:0;
            barrier(); winners[lane]+=value; barrier();
        }
        if(id<CELLS) material1[id]=uintBitsToFloat(winners[lane]-(flags!=0?1:0));
        if(lane==255) reduction[id/256]=double(winners[lane]);
        return;
    }
    if(pc.mode==13||pc.mode==14) {
        if(id>=3*CELLS) return;
        uint node=id%CELLS; int n=int(id/CELLS);
        if(floatBitsToUint(material0[node])==0) return;
        uint slot=uint(reduction[node/256])+floatBitsToUint(material1[node]);
        if(pc.mode==14) { if(n==0) ext[slot]=uintBitsToFloat(node); return; }
        ivec3 p=position(node);
        precise double value=0.0;
        uint kind=uint(aux[3]);
        if(aux[0]==0.0) {
            dvec3 v,i; lorentz(n,p,uint(aux[2]),v,i);
            value=kind<3?v[kind]:i[kind-3];
        } else if(kind==2) {
            precise double sum=0.0;
            for(uint pole=0;pole<uint(aux[1]);++pole) { dvec2 result; debye(n,p,pole,result); sum+=result.y; }
            value=1.0/(1.0+sum);
        } else { dvec2 result; debye(n,p,uint(aux[2]),result); value=result[kind]; }
        ext[uint(n)*pc.reserved+slot]=float(value);
        return;
    }
    if(pc.mode==15) {
        if(id>=4*pc.reserved) return;
        precise float phase=float(aux[0]*pc.dt);
        precise double distance=aux[1];
        ext[id]=id<2*pc.reserved?float((double(phase)-distance)/(double(phase)+distance)):float(double(phase)/distance);
        return;
    }
    if(pc.mode==16) {
        uint count=pc.reserved;
        if(id>=count) return;
        ivec3 start=ivec3(aux[0],aux[1],aux[2]),stop=ivec3(aux[3],aux[4],aux[5]),p=start;
        int n=int(aux[6]),a=(n+1)%3,b=(n+2)%3;
        uint nb=uint(stop[b]-start[b]+1),na=uint(stop[a]-start[a]+1);
        p[n]+=int(id/(na*nb)); p[a]+=int((id/nb)%na); p[b]+=int(id%nb);
        uint node=linear(p),cell=uint(n)*CELLS+node;
        for(uint k=1;k<10;++k) ext[id*10+k]=0.0;
        ext[id*10]=uintBitsToFloat(node); ext[id*10+6]=1.0;
        precise double omega=2.0*3.1415926535897932384626433832795lf*aux[14];
        precise double L=aux[8],R=aux[9],G=aux[10],C=aux[11],naturalC=double(ec0[cell]);
        if(aux[7]==0.0) {
            precise double cellC=C>0.0?C:naturalC;
            if(C>0.0) ec0[cell]=float(C);
            else {
                precise double Z=max(R,omega*L);
                if(1.0/(omega*cellC)<20.0*Z) { cellC=1.0/(omega*Z*20.0); ec0[cell]=float(cellC); }
            }
            ext[id*10+2]=float((pc.dt/cellC)/(1.0+pc.dt*G/(2.0*cellC)));
            ext[id*10+1]=aux[13]!=0.0?float(pc.dt/L):0.0;
            if(aux[12]!=0.0) ec1[cell]=float(G);
        } else {
            ec1[cell]=0.0;
            precise float cellC=ec0[cell];
            precise double reactance=omega*L-(C!=0.0?1.0/(C*omega):0.0);
            precise double Z=sqrt(R*R+reactance*reactance);
            if(1.0/(omega*double(cellC))<20.0*Z) { cellC=float(1.0/(omega*Z*20.0)); ec0[cell]=cellC; }
            precise double ib0=aux[15],b1=aux[16],b2=aux[17];
            ext[id*10+3]=float(0.5*pc.dt*ib0/double(cellC));
            ext[id*10+4]=float(0.5*pc.dt*(b1*ib0-1.0)/double(cellC));
            ext[id*10+5]=float(0.5*pc.dt*b2*ib0/double(cellC));
            ext[id*10+6]=float(1.0/(1.0+0.5*pc.dt*ib0/double(cellC)));
            ext[id*10+7]=float(ib0); ext[id*10+8]=float(b1); ext[id*10+9]=float(b2);
        }
        precise double cap=double(ec0[cell]),conductance=double(ec1[cell]);
        vv[cell]=cap>0.0?float((1.0-pc.dt*conductance/2.0/cap)/(1.0+pc.dt*conductance/2.0/cap)):0.0;
        vi[cell]=cap>0.0?float((pc.dt/cap)/(1.0+pc.dt*conductance/2.0/cap)):0.0;
        precise double inductance=double(ec2[cell]),resistance=double(ec3[cell]);
        ii[cell]=inductance>0.0?float((1.0-pc.dt*resistance/2.0/inductance)/(1.0+pc.dt*resistance/2.0/inductance)):0.0;
        iv[cell]=inductance>0.0?float((pc.dt/inductance)/(1.0+pc.dt*resistance/2.0/inductance)):0.0;
        return;
    }
    if(pc.mode==18) {
        uint count=pc.reserved;
        if(id>=4*count) return;
        uint part=id/count,node=id%count;
        bool voltage=part>=2; int component=int(part%2),axis=int(aux[0]),side=int(aux[1]);
        int a=(axis+1)%3,b=(axis+2)%3,incident=component==0?b:a,updated=component==0?a:b;
        ivec3 start=ivec3(aux[2],aux[3],aux[4]),stop=ivec3(aux[5],aux[6],aux[7]),p=start;
        p[a]+=int(node/uint(stop[b]-start[b]+1)); p[b]+=int(node%uint(stop[b]-start[b]+1));
        p[axis]=side==0?start[axis]-(voltage?1:0):stop[axis];
        dvec3 c;
        for(int n=0;n<3;++n) c[n]=g(n,p[n],(n==incident)!=voltage?1:0);
        precise double distance=0.0;
        for(int n=0;n<3;++n) distance+=abs((c[n]-aux[8+uint(n)])*aux[11+uint(n)]);
        precise double delay=distance*aux[21]/aux[20]/pc.dt+(voltage?1.0:0.0);
        precise float amplitude=float(aux[(voltage?17:14)+uint(incident)]*width(incident,p[incident],voltage));
        if(side==0) p[axis]+=voltage?1:-1;
        amplitude*=voltage?vi[uint(updated)*CELLS+linear(p)]:iv[uint(updated)*CELLS+linear(p)];
        if((voltage&&component==1&&side==0)||(voltage&&component==0&&side==1)||
            (!voltage&&component==0&&side==0)||(!voltage&&component==1&&side==1)) amplitude*=-1.0;
        if((p[a]==stop[a]&&component==(voltage?0:1))||(p[b]==stop[b]&&component==(voltage?1:0))) amplitude=0.0;
        ext[id*3]=uintBitsToFloat(uint(floor(delay))); ext[id*3+1]=float(delay-floor(delay)); ext[id*3+2]=amplitude;
        return;
    }
    if(pc.mode==10) {
        uint count=pc.reserved;
        if(id>=2*count) return;
        int axis=int(aux[0]),a=(axis+1)%3,b=(axis+2)%3;
        uint node=id%count, row=node/DIMS[b];
        ivec3 p=ivec3(0); p[axis]=int(aux[1]); p[a]=int(row); p[b]=int(node%DIMS[b]);
        dvec3 c=dvec3(g(0,p.x,0),g(1,p.y,0),g(2,p.z,0)); c[axis]=aux[3];
        uint prim=0;
        for(uint q=uint(aux[5+row]);q<uint(aux[6+row]);++q) {
            uint candidate=uint(aux[q]); if(inside(candidate,c)) { prim=candidate; break; }
        }
        int component=id<count?a:b;
        uint offset=prim*48+32+uint(component)*4;
        precise double epsilon=geo[offset],mu=geo[offset+2],distance=aux[4],phase;
        if(aux[2]>0.0) phase=aux[2]*pc.dt;
        else if(prim!=0) phase=299792458.0lf*pc.dt/sqrt(epsilon*mu);
        else phase=299792458.0lf/sqrt(epsilon*mu)*pc.dt;
        ext[id]=float((phase-distance)/(phase+distance));
        return;
    }
    if(pc.mode==11) {
        if(id>=pc.reserved) return;
        ext[id*6+3]=uintBitsToFloat(uint(aux[0]));
        ext[id*6+4]=uintBitsToFloat(uint(aux[1]));
        ext[id*6+5]=uintBitsToFloat(uint(aux[2]));
        return;
    }
    if(pc.mode==8) {
        if(id!=0) return;
        ivec3 start=ivec3(aux[0],aux[1],aux[2]),stop=ivec3(aux[3],aux[4],aux[5]);
        int n=int(aux[6]),a=(n+1)%3,b=(n+2)%3;
        precise double total=0.0;
        ivec3 p=start;
        for(p[n]=start[n];p[n]<stop[n];++p[n]) {
            precise double plane=0.0;
            for(p[a]=start[a];p[a]<=stop[a];++p[a]) for(p[b]=start[b];p[b]<=stop[b];++p[b])
                plane+=area(n,p,false)/width(n,p[n],false);
            total+=1.0/plane;
        }
        reduction[0]=1.0/total;
        return;
    }
    if(pc.mode==6) {
        if(id!=0) return;
        uint groups=(3*CELLS+255)/256;
        double sum=0.0;
        for(uint q=0;q<groups;++q) { double count=reduction[q]; reduction[q]=sum; sum+=count; }
        reduction[groups]=sum;
        return;
    }
    if(pc.mode==5||pc.mode==7) {
        uint lane=gl_LocalInvocationID.x;
        int n=int(id%3); uint node=id/3;
        ivec3 p=ivec3(node%DIMS.x,(node/DIMS.x)%DIMS.y,node/(DIMS.x*DIMS.y));
        bool magnetic=pc.reserved!=0;
        uint cell=uint(n)*CELLS+(id<3*CELLS?linear(p):0);
        if(pc.mode==7) {
            if(id>=3*CELLS) return;
            uint prim=floatBitsToUint(material0[id]); if(prim==0) return;
            uint slot=uint(reduction[id/256])+floatBitsToUint(material1[id]);
            precise double amplitude=geo[prim*48+32+uint(n)*4]*width(n,p[n],magnetic);
            ext[slot*6]=uintBitsToFloat(cell); ext[slot*6+1]=float(amplitude);
            ext[slot*6+2]=uintBitsToFloat(uint(geo[prim*48+30]/pc.dt));
            ext[slot*6+3]=0.0; ext[slot*6+4]=0.0; ext[slot*6+5]=0.0;
            return;
        }
        uint prim=0;
        if(id<3*CELLS) {
            bool valid=magnetic?all(lessThan(p,ivec3(DIMS)-1)):p[n]<int(DIMS[n])-1;
            if(valid) {
                dvec3 c;
                for(int axis=0;axis<3;++axis) c[axis]=g(axis,p[axis],(axis==n)!=magnetic?1:0);
                uint selected=sourceProperty(c,p);
                uint b=selected*48;
                bool type=magnetic?(geo[b+3]==2.0||geo[b+3]==3.0):(geo[b+3]==0.0||geo[b+3]==1.0);
                if(selected!=0&&geo[b+2]!=0.0&&geo[b+33+uint(n)*4]!=0.0&&type) {
                    if(geo[b+3]==1.0) { vv[cell]=0.0; vi[cell]=0.0; }
                    if(geo[b+3]==3.0) { ii[cell]=0.0; iv[cell]=0.0; }
                    if(geo[b+32+uint(n)*4]*width(n,p[n],magnetic)!=0.0) prim=selected;
                }
            }
            material0[id]=uintBitsToFloat(prim);
        }
        winners[lane]=prim!=0?1:0; barrier();
        // Inclusive workgroup scan, with an explicit read-before-write barrier.
        for(uint step=1;step<256;step*=2) {
            uint value=lane>=step?winners[lane-step]:0;
            barrier(); winners[lane]+=value; barrier();
        }
        if(id<3*CELLS) material1[id]=uintBitsToFloat(winners[lane]-(prim!=0?1:0));
        if(lane==255) reduction[id/256]=double(winners[lane]);
        return;
    }
    if(pc.mode==4) {
        uint count=pc.reserved;
        if(id>=3*count) return;
        int n=int(id/count),a=(n+1)%3,b=(n+2)%3;
        ivec3 dims=ivec3(aux[3],aux[4],aux[5]);
        ivec3 loc=ivec3(id%count/uint(dims.y*dims.z),(id%count/uint(dims.z))%uint(dims.y),id%count%uint(dims.z));
        ivec3 p=loc+ivec3(aux[0],aux[1],aux[2]);
        uint cell=uint(n)*CELLS+linear(p);
        uint stride=DIMS.x+DIMS.y+DIMS.z;
        precise dvec3 kv,ki;
        for(int axis=0;axis<3;++axis) {
            uint offset=axis==0?0:axis==1?DIMS.x:DIMS.x+DIMS.y;
            kv[axis]=aux[6+uint(n)*2*stride+offset+uint(p[axis])];
            ki[axis]=aux[6+(uint(n)*2+1)*stride+offset+uint(p[axis])];
        }
        precise dvec4 m=effective(n,p);
        precise double E=8.85418781762e-12lf;
        uint total=3*count;
        for(uint k=0;k<6;++k) ext[k*total+id]=0.0;
        if((kv.x+kv.y+kv.z)!=0.0&&m.y<1e3lf) {
            if((vv[cell]+vi[cell])!=0.0) {
                vv[cell]=float((2.0*E-kv[a]*pc.dt)/(2.0*E+kv[a]*pc.dt));
                vi[cell]=float((2.0*E*pc.dt)/(2.0*E+kv[a]*pc.dt)*width(n,p[n],false)/area(n,p,false));
                ext[id]=float((2.0*E-kv[b]*pc.dt)/(2.0*E+kv[b]*pc.dt));
                ext[total+id]=float((2.0*E-kv[n]*pc.dt)/(2.0*E+kv[b]*pc.dt)/m.x);
                ext[2*total+id]=float((2.0*E+kv[n]*pc.dt)/(2.0*E+kv[b]*pc.dt)/m.x);
            }
        } else {
            ext[id]=vv[cell]; vv[cell]=0.0; ext[2*total+id]=1.0;
        }
        if((ki.x+ki.y+ki.z)!=0.0) {
            if((ii[cell]+iv[cell])!=0.0) {
                ii[cell]=float((2.0*E-ki[a]*pc.dt)/(2.0*E+ki[a]*pc.dt));
                iv[cell]=float((2.0*E*pc.dt)/(2.0*E+ki[a]*pc.dt)*width(n,p[n],true)/area(n,p,true));
                ext[3*total+id]=float((2.0*E-ki[b]*pc.dt)/(2.0*E+ki[b]*pc.dt));
                ext[4*total+id]=float((2.0*E-ki[n]*pc.dt)/(2.0*E+ki[b]*pc.dt)/m.z);
                ext[5*total+id]=float((2.0*E+ki[n]*pc.dt)/(2.0*E+ki[b]*pc.dt)/m.z);
            }
        } else {
            ext[3*total+id]=ii[cell]; ii[cell]=0.0; ext[5*total+id]=1.0;
        }
        upmlIndices[pc.stores+id]=cell;
        upmlVoltage[pc.stores+id]=vec4(ext[id],ext[total+id],ext[2*total+id],0.0);
        upmlCurrent[pc.stores+id]=vec4(ext[3*total+id],ext[4*total+id],ext[5*total+id],0.0);
        return;
    }
    if(pc.mode==1) {
        uint lane=gl_LocalInvocationID.x;
        double value=1e200lf; uint order=0xffffffff;
        if(id<3*CELLS) {
            int n=int(id/CELLS); ivec3 p=position(id%CELLS);
            double v=dtValue(n,p); if(v>0.0&&v<value) value=v;
            order=uint(n)*CELLS+uint(p.z)*DIMS.y*DIMS.x+uint(p.y)*DIMS.x+uint(p.x);
        }
        minima[lane]=value; winners[lane]=order; barrier();
        for(uint step=128;step>0;step/=2) {
            if(lane<step&&(minima[lane+step]<minima[lane]||
                (minima[lane+step]==minima[lane]&&winners[lane+step]<winners[lane]))) {
                minima[lane]=minima[lane+step]; winners[lane]=winners[lane+step];
            } barrier();
        }
        if(lane==0) { uint group=(pc.base/256+gl_WorkGroupID.x)*2; reduction[group]=minima[0]; reduction[group+1]=double(winners[0]); }
        return;
    }
    if(id>=3*CELLS) return;
    int n=int(id/CELLS); ivec3 p=position(id%CELLS);
    if(pc.mode==0) {
        precise dvec4 m=effective(n,p);
        if(any(isnan(m))||any(isinf(m))) atomicOr(used[ERROR_INDEX],1);
        if((pc.stores&1)!=0) material0[id]=float(m.x);
        if((pc.stores&2)!=0) material1[id]=float(m.y);
        if((pc.stores&4)!=0) material2[id]=float(m.z);
        if((pc.stores&8)!=0) material3[id]=float(m.w);
        precise double length=width(n,p[n],false), A=area(n,p,false);
        ec0[id]=float(m.x*A/length); ec1[id]=float(m.y*A/length);
        length=width(n,p[n],true); A=area(n,p,true);
        ec2[id]=float(m.z*A/length); ec3[id]=float(m.w*A/length);
    } else if(pc.mode==2) {
        precise double C=double(ec0[id]),G=double(ec1[id]),L=double(ec2[id]),R=double(ec3[id]);
        vv[id]=C>0.0?float((1.0-pc.dt*G/2.0/C)/(1.0+pc.dt*G/2.0/C)):0.0;
        vi[id]=C>0.0?float((pc.dt/C)/(1.0+pc.dt*G/2.0/C)):0.0;
        ii[id]=L>0.0?float((1.0-pc.dt*R/2.0/L)/(1.0+pc.dt*R/2.0/L)):0.0;
        iv[id]=L>0.0?float((pc.dt/L)/(1.0+pc.dt*R/2.0/L)):0.0;
        for(int a=0;a<3;++a) {
            if((a!=n&&p[a]==0&&BC[2*a]!=-1)||(p[a]==int(DIMS[a])-1&&BC[2*a+1]!=-1)) { vv[id]=0.0; vi[id]=0.0; }
            if(p[a]==int(DIMS[a])-1||(p[a]==0&&BC[2*a]==1)||
                (a!=n&&p[a]==int(DIMS[a])-2&&BC[2*a+1]==1)) { ii[id]=0.0; iv[id]=0.0; }
        }
    } else if(pc.mode==9||pc.mode==17) {
        ivec3 start=ivec3(aux[0],aux[1],aux[2]),stop=ivec3(aux[3],aux[4],aux[5]);
        int direction=int(aux[6]);
        if(any(lessThan(p,start))||any(greaterThan(p,stop))) return;
        if(pc.mode==9&&n==direction&&p[n]<stop[n]) {
            if(aux[7]>0.0) ec0[id]=float(aux[9]*area(n,p,false)/width(n,p[n],false));
            if(aux[8]>0.0) ec1[id]=float(aux[10]*area(n,p,false)/width(n,p[n],false));
            if(aux[8]==0.0) { vv[id]=0.0; vi[id]=0.0; }
            else {
                precise double C=double(ec0[id]),G=double(ec1[id]);
                vv[id]=C>0.0?float((1.0-pc.dt*G/2.0/C)/(1.0+pc.dt*G/2.0/C)):0.0;
                vi[id]=C>0.0?float((pc.dt/C)/(1.0+pc.dt*G/2.0/C)):0.0;
            }
        }
        if(aux[11]!=0.0&&n!=direction&&(p[direction]==start[direction]||p[direction]==stop[direction])&&p[n]<stop[n]) {
            vv[id]=0.0; vi[id]=0.0; atomicAdd(used[PEC_INDEX+uint(n)],1);
        }
    } else if(pc.mode==3) {
        dvec3 c=dvec3(g(0,p.x,0),g(1,p.y,0),g(2,p.z,0)); c[n]=g(n,p[n],1);
        uint prim=property(c,p,1);
        if(geo[prim*48+1]==1.0) { vv[id]=0.0; vi[id]=0.0; atomicAdd(used[PEC_INDEX+uint(n)],1); }
    }
}
)glsl" };
#endif
