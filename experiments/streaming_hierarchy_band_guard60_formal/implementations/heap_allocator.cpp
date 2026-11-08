#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <vector>

// Exact executable-action heap. No sharing-pattern or attention-output enumeration.
// Stable ties: K sharing, V sharing, pruning; then token/layer/head birth index.
struct Action { double cost; int64_t root; int32_t epoch; uint8_t kind; };
struct Greater {
 bool operator()(const Action& a,const Action& b) const {
  if(a.cost!=b.cost)return a.cost>b.cost;
  if(a.kind!=b.kind)return a.kind>b.kind;
  return a.root>b.root;
 }
};
using Clock=std::chrono::steady_clock;
extern "C" int64_t allocate_heap(int64_t M,int64_t LH,int64_t N,int64_t budget,
 const float* importance,const float* costs,const uint8_t* order,uint8_t* stage,
 uint8_t* alive,int allow_prune,uint64_t* counters,uint64_t* by_head,
 double* timings,double* examples,int64_t examples_capacity) {
 try {
  auto begin=Clock::now();
  std::vector<Action> heap;heap.reserve(3*M);
  std::vector<int32_t> epoch(M,0);
  int64_t total=0,recent=(N+4)/5,example_count=0;
  auto add=[&](int64_t r,bool ordered) {
   if(!alive[r])return;
   double I=importance[r],remaining=2.;int slots=2;
   auto push=[&](double cost,uint8_t kind) {
    heap.push_back({cost,r,epoch[r],kind});++counters[4];
    if(ordered)std::push_heap(heap.begin(),heap.end(),Greater());
    counters[7]=std::max(counters[7],uint64_t(heap.capacity()*sizeof(Action)+epoch.capacity()*sizeof(int32_t)));
   };
   for(int k=0;k<2;++k){
    int s=stage[2*r+k];slots+=3-s;
    for(int j=s;j<3;++j)remaining+=std::min(double(costs[6*r+3*k+j]),1.);
    if(s<3)push(I*std::min(double(costs[6*r+3*k+s]),1.),uint8_t(k));
   }
   if(allow_prune && r/LH < N-recent)push(I*remaining/slots,2);
  };
  for(int64_t r=0;r<M;++r){
   if(!std::isfinite(importance[r]) || importance[r]<0)return -2;
   for(int k=0;k<2;++k)if(stage[2*r+k]>3)return -3;
   if(alive[r]){total+=8-stage[2*r]-stage[2*r+1];add(r,false);}
  }
  auto scored=Clock::now();timings[0]=std::chrono::duration<double>(scored-begin).count();
  std::make_heap(heap.begin(),heap.end(),Greater());
  while(total>budget){
   if(heap.empty())return -4;
   std::pop_heap(heap.begin(),heap.end(),Greater());Action a=heap.back();heap.pop_back();++counters[5];
   int64_t r=a.root;
   if(!alive[r] || a.epoch!=epoch[r]){++counters[6];continue;}
   int before=8-stage[2*r]-stage[2*r+1],released=1;
   int kstage=stage[2*r],vstage=stage[2*r+1];
   if(a.kind<2){
    int k=a.kind,j=stage[2*r+k];
    ++counters[9+3*k+order[6*r+3*k+j]];
    ++stage[2*r+k];++counters[k];
   }else{
    if(r/LH>=N-recent)return -5;
    released=before;alive[r]=0;++counters[2];
    if(before>2)++counters[8];
   }
   ++by_head[3*(r%LH)+a.kind];total-=released;counters[3]+=released;++epoch[r];
   if(example_count<examples_capacity){
    double* e=examples+8*example_count++;
    e[0]=double(r);e[1]=a.kind;e[2]=a.cost;e[3]=before;e[4]=released;e[5]=kstage;e[6]=vstage;e[7]=double(total);
   }
   add(r,true);
  }
  counters[15]=example_count;
  timings[1]=std::chrono::duration<double>(Clock::now()-scored).count();
  return total;
 } catch(...) { return -9; }
}
