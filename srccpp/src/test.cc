#include "cnpy.h"
#include <complex>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <iostream>
#include <Fastor/Fastor.h> 
#include <math.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xslice.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xsort.hpp>

template<typename T>
void show_vector(std::vector<T> values)
{
    std::cout<<"********the show **********"<<std::endl;
    for(int i=0;i<values.size();i++)
        {
            std::cout<<values[i]<<" ";
        }
    std::cout<<"********the show **********"<<std::endl;
}




Fastor::Tensor<float,1,80,240> np_max_pool2d_80_240(Fastor::Tensor<float,1,80,240> mat, std::vector<int> kernel, int stride, int padding)
{

int mat_c=1; 
int mat_h=80; 
int mat_w=240;  

//mat = np.pad(mat, ((0, 0), (padding, padding), (padding, padding)), 'constant')
//拷贝的方式来做
std::cout<<"in np_max_pool2d"<<std::endl;
//print("the mat",mat);
std::cout<<"in np_max_pool2d"<<std::endl;

Fastor::Tensor<float,1,80+2,240+2>  mat_zero;
mat_zero.zeros();
mat_zero(Fastor::all,Fastor::seq(1,81),Fastor::seq(1,241))=mat;

std::cout<<" mat_zero"<<std::endl;
//print("mat_zero",mat_zero);
std::cout<<" mat_zero"<<std::endl;



int new_h = floor(float(mat_h + 2 * padding - (kernel[1] - 1) - 1) / float(stride)) + 1;
int new_w = floor(float(mat_w + 2 * padding - (kernel[0] - 1) - 1) / float(stride)) + 1;

std::cout<<"new_h "<<new_h<<" new_w "<<new_w<<std::endl;

//result = np.zeros((mat_c, new_h, new_w))

Fastor::Tensor<float,1,80,240> result;
result.zeros();

    // python code
    // for c_idx in range(mat_c):
    //     #print("c_idx ",c_idx)
    //     for h_idx in range(0, new_h, stride):
    //         for w_idx in range(0, new_w, stride):
    //             temp = np.max(mat[c_idx][h_idx:h_idx + kernel[0], w_idx:w_idx + kernel[1]])#某个维度上的最大值
    //             result[c_idx][h_idx][w_idx] = temp

for(int c_idx=0;c_idx<mat_c;c_idx++)
{
    for(int h_idx=0;h_idx<new_h;h_idx+=stride)
    {
        for(int w_idx=0;w_idx<new_w;w_idx+=stride)
        {
            float temp=Fastor::max(mat_zero(c_idx,Fastor::seq(h_idx,h_idx+kernel[0]),Fastor::seq(w_idx,w_idx+kernel[1])));
            result(c_idx,h_idx,w_idx)=temp;
        }
    }
}

return result;
}


Fastor::Tensor<float,1,160,480> np_max_pool2d_160_480(Fastor::Tensor<float,1,160,480> mat, std::vector<int> kernel, int stride, int padding)
{

int mat_c=1; 
int mat_h=160; 
int mat_w=480;  

//mat = np.pad(mat, ((0, 0), (padding, padding), (padding, padding)), 'constant')
//拷贝的方式来做
std::cout<<"in np_max_pool2d"<<std::endl;
//print("the mat",mat);
std::cout<<"in np_max_pool2d"<<std::endl;

Fastor::Tensor<float,1,160+2,480+2>  mat_zero;
mat_zero.zeros();
mat_zero(Fastor::all,Fastor::seq(1,161),Fastor::seq(1,481))=mat;

std::cout<<" mat_zero"<<std::endl;
//print("mat_zero",mat_zero);
std::cout<<" mat_zero"<<std::endl;



int new_h = floor(float(mat_h + 2 * padding - (kernel[1] - 1) - 1) / float(stride)) + 1;
int new_w = floor(float(mat_w + 2 * padding - (kernel[0] - 1) - 1) / float(stride)) + 1;

std::cout<<"new_h "<<new_h<<" new_w "<<new_w<<std::endl;

//result = np.zeros((mat_c, new_h, new_w))

Fastor::Tensor<float,1,160,480> result;
result.zeros();

    // python code
    // for c_idx in range(mat_c):
    //     #print("c_idx ",c_idx)
    //     for h_idx in range(0, new_h, stride):
    //         for w_idx in range(0, new_w, stride):
    //             temp = np.max(mat[c_idx][h_idx:h_idx + kernel[0], w_idx:w_idx + kernel[1]])#某个维度上的最大值
    //             result[c_idx][h_idx][w_idx] = temp

for(int c_idx=0;c_idx<mat_c;c_idx++)
{
    for(int h_idx=0;h_idx<new_h;h_idx+=stride)
    {
        for(int w_idx=0;w_idx<new_w;w_idx+=stride)
        {
            float temp=Fastor::max(mat_zero(c_idx,Fastor::seq(h_idx,h_idx+kernel[0]),Fastor::seq(w_idx,w_idx+kernel[1])));
            result(c_idx,h_idx,w_idx)=temp;
        }
    }
}

return result;
}



Fastor::Tensor<float,1,48,144> np_max_pool2d_48_144(Fastor::Tensor<float,1,48,144> mat, std::vector<int> kernel, int stride, int padding)
{

int mat_c=1; 
int mat_h=48; 
int mat_w=144;  

//mat = np.pad(mat, ((0, 0), (padding, padding), (padding, padding)), 'constant')
//拷贝的方式来做
std::cout<<"in np_max_pool2d"<<std::endl;
//print("the mat",mat);
std::cout<<"in np_max_pool2d"<<std::endl;

Fastor::Tensor<float,1,48+2,144+2>  mat_zero;
mat_zero.zeros();
mat_zero(Fastor::all,Fastor::seq(1,49),Fastor::seq(1,145))=mat;

std::cout<<" mat_zero"<<std::endl;
//print("mat_zero",mat_zero);
std::cout<<" mat_zero"<<std::endl;



int new_h = floor(float(mat_h + 2 * padding - (kernel[1] - 1) - 1) / float(stride)) + 1;
int new_w = floor(float(mat_w + 2 * padding - (kernel[0] - 1) - 1) / float(stride)) + 1;

std::cout<<"new_h "<<new_h<<" new_w "<<new_w<<std::endl;

//result = np.zeros((mat_c, new_h, new_w))

Fastor::Tensor<float,1,48,144> result;
result.zeros();

    // python code
    // for c_idx in range(mat_c):
    //     #print("c_idx ",c_idx)
    //     for h_idx in range(0, new_h, stride):
    //         for w_idx in range(0, new_w, stride):
    //             temp = np.max(mat[c_idx][h_idx:h_idx + kernel[0], w_idx:w_idx + kernel[1]])#某个维度上的最大值
    //             result[c_idx][h_idx][w_idx] = temp

for(int c_idx=0;c_idx<mat_c;c_idx++)
{
    for(int h_idx=0;h_idx<new_h;h_idx+=stride)
    {
        for(int w_idx=0;w_idx<new_w;w_idx+=stride)
        {
            float temp=Fastor::max(mat_zero(c_idx,Fastor::seq(h_idx,h_idx+kernel[0]),Fastor::seq(w_idx,w_idx+kernel[1])));
            result(c_idx,h_idx,w_idx)=temp;
        }
    }
}

return result;
}

//python code
// def np_topk(mat, topk_n):
//     topk_data_sort = -np.sort(-mat)[:topk_n]#sort on rows
//     topk_index_sort = np.argsort(-mat)[:topk_n]#sort return position np.array([3, 1, 2]) #[1 2 0]
//     return topk_data_sort, topk_index_sort



template<typename T>
std::vector<int> sort_index(std::vector<T> data)
{
    //std::vector<T> data = {5, 16, 4, 7};   
    std::vector<int> index(data.size(), 0);
    for (int i = 0 ; i != index.size() ; i++) {
        index[i] = i;
    }
    std::sort(index.begin(), index.end(),
        [&](const int & a, const int & b) {
            return (data[a] < data[b]);
        }
    );
    // for (T i = 0 ; i != index.size() ; i++) {
    //     std::cout << index[i] << std::endl;
    // }
    return index;
}



struct n_topk_res
{
    std::vector<float> topk_data_sort;
    std::vector<int> top_index_sort;
};



n_topk_res np_topk_160_480(Fastor::Tensor<float,160*480,1> mat,int topk_n)
{
std::cout<<"##################in np_topk##############"<<std::endl;
Fastor::Tensor<float,160*480,1> mat_oppo_=-mat;
std::vector<float> mat_vector;
for(int i=0;i<160*480;i++)
{
    mat_vector.push_back(mat_oppo_(i,0));
    std::cout<<"the mat_oppo_"<<mat_oppo_(i,0)<<std::endl;
}
    std::cout<<"topk_n"<<topk_n<<std::endl;
    std::cout<<"###############in np_topk##############"<<std::endl;
    std::vector<int> topk_index_=sort_index(mat_vector);
    std::vector<float> topk_data_;
    for(int index=0;index<topk_index_.size();index++)
    {
        topk_data_.push_back(-mat_vector[topk_index_[index]]);
    }



    std::vector<int>::const_iterator first1 = topk_index_.begin();
    std::vector<int>::const_iterator last1  = topk_index_.begin() + topk_n;
    std::vector<int> top_index_sort(first1, last1);

    std::vector<float>::const_iterator first2 = topk_data_.begin();
    std::vector<float>::const_iterator last2  = topk_data_.begin() + topk_n;
    std::vector<float> top_data_sort(first2, last2);



    n_topk_res temp;
    temp.top_index_sort=top_index_sort;
    temp.topk_data_sort=top_data_sort;
    return temp;

}



n_topk_res np_topk_80_240(Fastor::Tensor<float,80*240,1> mat,int topk_n)
{
std::cout<<"##################in np_topk##############"<<std::endl;
Fastor::Tensor<float,80*240,1> mat_oppo_=-mat;
std::vector<float> mat_vector;
for(int i=0;i<80*240;i++)
{
    mat_vector.push_back(mat_oppo_(i,0));
    std::cout<<"the mat_oppo_"<<mat_oppo_(i,0)<<std::endl;
}
    std::cout<<"topk_n"<<topk_n<<std::endl;
    std::cout<<"###############in np_topk##############"<<std::endl;
    std::vector<int> topk_index_=sort_index(mat_vector);
    std::vector<float> topk_data_;
    for(int index=0;index<topk_index_.size();index++)
    {
        topk_data_.push_back(-mat_vector[topk_index_[index]]);
    }



    std::vector<int>::const_iterator first1 = topk_index_.begin();
    std::vector<int>::const_iterator last1  = topk_index_.begin() + topk_n;
    std::vector<int> top_index_sort(first1, last1);

    std::vector<float>::const_iterator first2 = topk_data_.begin();
    std::vector<float>::const_iterator last2  = topk_data_.begin() + topk_n;
    std::vector<float> top_data_sort(first2, last2);



    n_topk_res temp;
    temp.top_index_sort=top_index_sort;
    temp.topk_data_sort=top_data_sort;
    return temp;

}

n_topk_res np_topk_48_144(Fastor::Tensor<float,48*144,1> mat,int topk_n)
{
    std::cout<<"###############in np_topk##############"<<std::endl;
    Fastor::Tensor<float,48*144,1> mat_oppo_= -mat;
    std::vector<float> mat_vector;
    for(int i=0;i<48*144;i++)
    {
        mat_vector.push_back(mat_oppo_(i,0));
        std::cout<<"the mat_oppo_ "<<mat_oppo_(i,0)<<std::endl;
    }
    std::cout<<"topk_n"<<topk_n<<std::endl;
    std::cout<<"###############in np_topk##############"<<std::endl;
    std::vector<int> topk_index_=sort_index(mat_vector);
    std::vector<float> topk_data_;
    for(int index=0;index<topk_index_.size();index++)
    {
        topk_data_.push_back(-mat_vector[topk_index_[index]]);
    }



    std::vector<int>::const_iterator first1 = topk_index_.begin();
    std::vector<int>::const_iterator last1  = topk_index_.begin() + topk_n;
    std::vector<int> top_index_sort(first1, last1);

    std::vector<float>::const_iterator first2 = topk_data_.begin();
    std::vector<float>::const_iterator last2  = topk_data_.begin() + topk_n;
    std::vector<float> top_data_sort(first2, last2);



    n_topk_res temp;
    temp.top_index_sort=top_index_sort;
    temp.topk_data_sort=top_data_sort;
    return temp;
}




struct decode_lines_data
{
std::vector<std::vector<float>> np_center_ptss_v;
std::vector<std::vector<float>> np_lines_v;
std::vector<float> np_scores_finall;
};




decode_lines_data deccode_lines_80_240(Fastor::Tensor<float,1,9,80,240> np_tpMap, float score_thresh=0.1, float len_thresh=2.0, float topk_n=100.0, float ksize=3.0)
{
//(1, 4, 80, 240)
//np_b, np_c, np_h, np_w = np_tpMap.shape# (1, 9, 80, 240)

int np_b = 1;
int np_c = 4;
int np_h = 80;
int np_w = 240;

Fastor::Tensor<float,1,4,80,240> np_displacement = np_tpMap(Fastor::all, Fastor::seq(1,5), Fastor::all, Fastor::all);

std::cout<<"run 1"<<std::endl;

//print("np_displacement",np_displacement);


std::cout<<"run 1"<<std::endl;

//np_center = np_tpMap[:, 0, :, :]
Fastor::Tensor<float,1,80,240> np_center = np_tpMap(Fastor::all,0,Fastor::all,Fastor::all);


std::cout<<"run 2"<<std::endl;
//print("np_center",np_center);
std::cout<<"run 2"<<std::endl;

Fastor::Tensor<float,1,80,240> np_center_oppo = -np_center;

std::cout<<"run 3"<<std::endl;
//print("np_center_oppo",np_center_oppo);
std::cout<<"run 3"<<std::endl;

// np_heat = 1 / (1 + np.exp(-np_center))

Fastor::Tensor<float,1,80,240> np_heat = 1.0/(1.0+exp(np_center_oppo));

std::cout<<"run 4"<<std::endl;
//print("np_heat",np_heat);

std::cout<<"run 4"<<std::endl;

std::vector<int> ksize_={int(ksize),int(ksize)};

std::cout<<"run 5"<<std::endl;

Fastor::Tensor<float,1,80,240> np_hmax = np_max_pool2d_80_240(np_heat, ksize_, 1, 1);



std::cout<<"run 6"<<std::endl;
//print("np_hmax",np_hmax);
std::cout<<"run 6"<<std::endl;

//python code
//np_keep = (np_hmax == np_heat).astype('float32')
Fastor::Tensor<bool,1,80,240> np_keep = np_hmax==np_heat;

std::cout<<"run 7"<<std::endl;
//print("np_keep",np_keep);
std::cout<<"run 7"<<std::endl;

Fastor::Tensor<float,1,80,240> np_heat_new = np_heat(np_keep)*1;

std::cout<<"run 8"<<std::endl;
//print("np_heat_new",np_heat_new);
std::cout<<"run 8"<<std::endl;

//python code
//np_heat = np_heat.reshape(-1, )#the same as flatten

Fastor::Tensor<float,80*240,1> np_heat_flatten = Fastor::flatten(np_heat_new);

std::cout<<"run 9"<<std::endl;
//print("np_heat_flatten",np_heat_flatten);
std::cout<<"run 9"<<std::endl;

//python code
//np_heat = np.where(np_heat < score_thresh, np.zeros_like(np_heat), np_heat)

Fastor::Tensor<float,80*240,1> score_thresh_matrix;

std::cout<<"run 10"<<std::endl;

score_thresh_matrix.ones();

std::cout<<"run 11"<<std::endl;

score_thresh_matrix=score_thresh_matrix*score_thresh;

std::cout<<"run 12"<<std::endl;
//print("score_thresh_matrix",score_thresh_matrix);
std::cout<<"run 12"<<std::endl;

Fastor::Tensor<bool,80*240,1> np_heat_mask1= score_thresh_matrix<np_heat_flatten;

std::cout<<"run 13"<<std::endl;

//print("np_heat_mask1",np_heat_mask1);
std::cout<<"run 13"<<std::endl;

Fastor::Tensor<float,80*240,1> np_heat_after_choice=np_heat_flatten(np_heat_mask1)*1;


std::cout<<"run 14"<<std::endl;
//print("np_heat_after_choice",np_heat_after_choice);
std::cout<<"run 14"<<std::endl;


//python code

//np_scores, np_indices = np_topk(np_heat, topk_n)
//np_valid_inx = np.where(np_scores > score_thresh)

// (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
//        17, 18, 19], dtype=int64),)

n_topk_res np_scores_with_np_indices =np_topk_80_240(np_heat_after_choice,topk_n);

//std::vector<float> ksd={-2.23,-1.63,-1.36,-1.68,-25.369,-5.954,-0.0};
//std::vector<int> sort_indexs=sort_index(ksd);
//show_vector(sort_indexs);


std::cout<<"run 15"<<std::endl;
std::vector<float> data_sort=np_scores_with_np_indices.topk_data_sort;
std::vector<int> index_sort=np_scores_with_np_indices.top_index_sort;

//show_vector(data_sort);

std::cout<<"\n";

//show_vector(index_sort);



std::cout<<"run 15"<<std::endl;

const size_t top_c = (size_t)topk_n;  
std::cout<<"run 16"<<std::endl;

Fastor::Tensor<float,150,1> np_scores(np_scores_with_np_indices.topk_data_sort);
std::cout<<"run 17"<<std::endl;
//print("np_scores",np_scores);
std::cout<<"run 17"<<std::endl;


Fastor::Tensor<int,150,1> np_indices(np_scores_with_np_indices.top_index_sort);
std::cout<<"run 18"<<std::endl;

Fastor::Tensor<bool,150,1> np_scores_after_mask = np_scores>score_thresh;
std::cout<<"run 19"<<std::endl;


//get the 1 or true 位置

std::vector<int> np_valid_inx_vector;
std::cout<<"run 20"<<std::endl;


for(int index_valid=0;index_valid<150;index_valid++)
{
    if(np_scores_after_mask(index_valid,0)==1)
    {
        np_valid_inx_vector.push_back(index_valid);
    }
}

std::cout<<"run 21"<<std::endl;
//show_vector(np_valid_inx_vector);
std::cout<<"run 21"<<std::endl;

//python code
//   np_scores = np_scores[np_valid_inx]
//     np_indices = np_indices[np_valid_inx]
//     print("\n")
//     #print(np_scores)

//     print("\n")
//     #print(np_indices)
//     #simple


std::vector<float> np_scores_after_valid;
std::vector<int> np_indices_after_valid;
std::cout<<"run 22"<<std::endl;

for(int index_valid_=0;index_valid_<np_valid_inx_vector.size();index_valid_++)
{
    np_scores_after_valid.push_back(np_scores(index_valid_,0));
    np_indices_after_valid.push_back(np_indices(index_valid_,0));
}
std::cout<<"run 23"<<std::endl;

//show_vector(np_scores_after_valid);

//show_vector(np_indices_after_valid);



//python code
//     np_yy = np.floor_divide(np_indices, np_w)[:, np.newaxis]#every element divide add a dim
//     print("\n")
//     print(np_yy)


//     np_xx = np.fmod(np_indices, np_w)[:, np.newaxis]#every element % operator add a dim
//     print("\n")
//     print(np_xx)


std::vector<float> np_xx;
std::vector<float> np_yy;

std::cout<<"run 24 "<<np_indices_after_valid.size()<<std::endl;

for(int index_np_indices=0;index_np_indices<np_indices_after_valid.size();index_np_indices++)
{
    std::cout<<" the half left "<<np_indices_after_valid[index_np_indices]<<std::endl;
    np_yy.push_back(float(np_indices_after_valid[index_np_indices])/float(np_w));
    np_xx.push_back(float(int(np_indices_after_valid[index_np_indices]%int(np_w))));
}

std::cout<<"run 25"<<std::endl;

show_vector(np_yy);

for(int yy_index=0;yy_index<np_yy.size();yy_index++)
{
    np_yy[yy_index]=(int)(np_yy[yy_index]);
}


show_vector(np_yy);

printf("\n");

show_vector(np_xx);

std::cout<<"run 25"<<std::endl;

Fastor::Tensor<float,20,1> np_xx_tensor(np_xx);
Fastor::Tensor<float,20,1> np_yy_tensor(np_yy);

std::cout<<"run 26"<<std::endl;




//     np_center_ptss = np.concatenate((np_xx, np_yy), axis=-1)

//     print("\n")
//     print(np_center_ptss) (20,2)

//     #depends on column 
Fastor::Tensor<float,20,2> np_center_ptss;
np_center_ptss.ones();

std::cout<<"run 27"<<std::endl;

np_center_ptss(Fastor::all,Fastor::seq(0,1))=np_xx_tensor;
np_center_ptss(Fastor::all,Fastor::seq(1,2))=np_yy_tensor;


print("np_center_ptss",np_center_ptss);


std::cout<<"run 28"<<std::endl;





//     np_start_point = np_center_ptss + np.squeeze(np_displacement[0, :2, np_yy, np_xx])//(20, 1, 2)
//     print("\n")
//     print(np_start_point)


Fastor::Tensor<float,20,2> np_displacement_temp_s;
np_displacement_temp_s.ones();

std::cout<<"run 29"<<std::endl;


for(int index_dim1=0;index_dim1<2;index_dim1++)
{
for(int index_np=0;index_np<20;index_np++)
{
    np_displacement_temp_s(index_np,index_dim1)=np_displacement(0,index_dim1,int(np_yy[index_np]),int(np_xx[index_np]));
}
}

std::cout<<"run 30"<<std::endl;



Fastor::Tensor<float,20,2> np_start_point = np_center_ptss+np_displacement_temp_s;

std::cout<<"run 31"<<std::endl;
//print("np_start_point",np_start_point);
std::cout<<"run 31"<<std::endl;

    
//     np_end_point = np_center_ptss + np.squeeze(np_displacement[0, 2:, np_yy, np_xx])
//     print("\n")
//     print(np_end_point)



Fastor::Tensor<float,20,2> np_displacement_temp_e;
np_displacement_temp_e.ones();

std::cout<<"run 32"<<std::endl;


for(int index_dim2=2;index_dim2<4;index_dim2++)
{
    for(int index_np2=0;index_np2<20;index_np2++)
    {
        np_displacement_temp_e(index_np2,index_dim2-2)=np_displacement(0,index_dim2,int(np_yy[index_np2]),int(np_xx[index_np2]));
    }
}

std::cout<<"run 33"<<std::endl;


Fastor::Tensor<float,20,2> np_end_point=np_center_ptss+np_displacement_temp_e;

std::cout<<"run 34"<<std::endl;
print("np_end_point",np_end_point);
std::cout<<"run 34"<<std::endl;


//     np_lines = np.concatenate((np_start_point, np_end_point), axis=-1)
//     print("\n")
//     print(np_lines)
       
//(20,2 20,2)

Fastor::Tensor<float,20,4> np_lines;
np_lines.ones();

std::cout<<"run 35"<<std::endl;

np_lines(Fastor::all,Fastor::seq(0,2))=np_start_point;
np_lines(Fastor::all,Fastor::seq(2,4))=np_end_point;

std::cout<<"run 36"<<std::endl;
//print("np_lines",np_lines);
std::cout<<"run 36"<<std::endl;




//     np_all_lens = (np_end_point - np_start_point) ** 2
//     print("\n")
//     print(np_all_lens)
Fastor::Tensor<float,20,2> np_all_lens = np_end_point-np_start_point;
std::cout<<"run 37"<<std::endl;


for(int index_dim_c=0;index_dim_c<20;index_dim_c++)
{
    for(int index_dim_r=0;index_dim_r<2;index_dim_r++)
    {
        np_all_lens(index_dim_c,index_dim_r)=np_all_lens(index_dim_c,index_dim_r)*np_all_lens(index_dim_c,index_dim_r);
    }
}

std::cout<<"run 38"<<std::endl;
//print("np_all_lens",np_all_lens);
std::cout<<"run 38"<<std::endl;

//     np_all_lens = np_all_lens.sum(axis=-1)# rows add
//     print("\n")
//     print(np_all_lens) (20,1)

//

Fastor::Tensor<float,20,1> np_all_len_temp;
np_all_len_temp.zeros();
std::cout<<"run 39"<<std::endl;


for(int index_colums=0;index_colums<2;index_colums++)
{
    np_all_len_temp(Fastor::all,Fastor::seq(0,1)) += np_all_lens(Fastor::all,Fastor::seq(index_colums,index_colums+1));
}
std::cout<<"run 40"<<std::endl;

//print("np_all_len_temp",np_all_len_temp);

std::cout<<"run 40"<<std::endl;



//     np_all_lens = np.sqrt(np_all_lens)# every element sqrt
//     print("\n")
//     print(np_all_lens) 
// (20,1)


Fastor::Tensor<float,20,1> np_all_lens_temp_sqrt;
np_all_lens_temp_sqrt.ones();

std::cout<<"run 41"<<std::endl;

for(int index_rows_temp=0;index_rows_temp<20;index_rows_temp++)
{
    np_all_lens_temp_sqrt(index_rows_temp,0)=std:: sqrt(np_all_len_temp(index_rows_temp,0));
}

std::cout<<"run 42"<<std::endl;
//print("np_all_lens_temp_sqrt",np_all_lens_temp_sqrt);

std::cout<<"run 42"<<std::endl;




//     np_valid_inx = np.where(np_all_lens > len_thresh)
//     print("\n")
//     print("\n")
//     print(np_valid_inx)
Fastor::Tensor<float,20,1> len_thresh_matrix;
Fastor::Tensor<float,20,1> len_mask_temp=len_thresh_matrix*len_thresh;
Fastor::Tensor<bool,20,1> np_all_lens_m = np_all_lens_temp_sqrt>len_mask_temp;
std::cout<<"run 43"<<std::endl;


std::vector<int> np_valid_inx;

for(int index_np_all_lens_m=0;index_np_all_lens_m<20;index_np_all_lens_m++)
{
    if(np_all_lens_m(index_np_all_lens_m,0)==1)
    {
        np_valid_inx.push_back(index_np_all_lens_m);
    }
}

std::cout<<"run 44"<<std::endl;
show_vector(np_valid_inx);

std::cout<<"run 44"<<std::endl;


//     np_center_ptss = np_center_ptss[np_valid_inx]
//     print("\n")
//     print(np_center_ptss) (20,2)


std::vector<std::vector<float>> np_center_ptss_v;
for(int row_np_center_ptss=0;row_np_center_ptss<np_valid_inx.size();row_np_center_ptss++)
{
std::vector<float> rows_all;
for(int colum_np_center_ptss=0;colum_np_center_ptss<2;colum_np_center_ptss++)
{
rows_all.push_back(np_center_ptss(row_np_center_ptss,colum_np_center_ptss));
}
np_center_ptss_v.push_back(rows_all);
}
std::cout<<"run 45"<<std::endl;

for(int i_=0;i_<np_center_ptss_v.size();i_++)
{
for(int j_=0;j_<np_center_ptss_v[i_].size();j_++)
{
    std::cout<<np_center_ptss_v[i_][j_]<<" ";
}
std::cout<<"\n";
}

std::cout<<"run 45"<<std::endl;


//     np_lines = np_lines[np_valid_inx] (20,4)
//     print("\n")
//     print(np_lines)
std::vector<std::vector<float>> np_lines_v;
for(int row_np_lines_v=0;row_np_lines_v<np_valid_inx.size();row_np_lines_v++)
{
    std::vector<float> rows_all_t;
    for(int colum_np_lines_v=0;colum_np_lines_v<4;colum_np_lines_v++)
    {
        rows_all_t.push_back(np_lines(row_np_lines_v,colum_np_lines_v));
    }
    np_lines_v.push_back(rows_all_t);
}

std::cout<<"run 46"<<std::endl;
for(int i_=0;i_<np_lines_v.size();i_++)
{
for(int j_=0;j_<np_lines_v[i_].size();j_++)
{
    std::cout<<np_lines_v[i_][j_]<<" ";
}
std::cout<<"\n";
}
std::cout<<"run 46"<<std::endl;



    
//     np_scores = np_scores[np_valid_inx] (20,1)
//     print("\n")
//     print(np_scores)

std::vector<float> np_scores_finall;

for(int index_scores=0;index_scores<np_valid_inx.size();index_scores++)
{
    np_scores_finall.push_back(np_scores_after_valid[index_scores]);
}

std::cout<<"run 47"<<std::endl;
show_vector(np_scores_finall);
std::cout<<"run 47"<<std::endl;

decode_lines_data temp_re;
temp_re.np_scores_finall=np_scores_finall;
temp_re.np_lines_v=np_lines_v;
temp_re.np_center_ptss_v=np_center_ptss_v;

return temp_re;

}



decode_lines_data deccode_lines_160_480(Fastor::Tensor<float,1,9,160,480> np_tpMap, float score_thresh=0.1, float len_thresh=2.0, float topk_n=100.0, float ksize=3.0)
{
//(1, 4, 80, 240)
//np_b, np_c, np_h, np_w = np_tpMap.shape# (1, 9, 80, 240)

int np_b = 1;
int np_c = 4;
int np_h = 160;
int np_w = 480;

Fastor::Tensor<float,1,4,160,480> np_displacement = np_tpMap(Fastor::all, Fastor::seq(1,5), Fastor::all, Fastor::all);

std::cout<<"run 1"<<std::endl;

//print("np_displacement",np_displacement);


std::cout<<"run 1"<<std::endl;

//np_center = np_tpMap[:, 0, :, :]
Fastor::Tensor<float,1,160,480> np_center = np_tpMap(Fastor::all,0,Fastor::all,Fastor::all);


std::cout<<"run 2"<<std::endl;
//print("np_center",np_center);
std::cout<<"run 2"<<std::endl;

Fastor::Tensor<float,1,160,480> np_center_oppo = -np_center;

std::cout<<"run 3"<<std::endl;
//print("np_center_oppo",np_center_oppo);
std::cout<<"run 3"<<std::endl;

// np_heat = 1 / (1 + np.exp(-np_center))

Fastor::Tensor<float,1,160,480> np_heat = 1.0/(1.0+exp(np_center_oppo));

std::cout<<"run 4"<<std::endl;
//print("np_heat",np_heat);

std::cout<<"run 4"<<std::endl;

std::vector<int> ksize_={int(ksize),int(ksize)};

std::cout<<"run 5"<<std::endl;

Fastor::Tensor<float,1,160,480> np_hmax = np_max_pool2d_160_480(np_heat, ksize_, 1, 1);



std::cout<<"run 6"<<std::endl;
//print("np_hmax",np_hmax);
std::cout<<"run 6"<<std::endl;

//python code
//np_keep = (np_hmax == np_heat).astype('float32')
Fastor::Tensor<bool,1,160,480> np_keep = np_hmax==np_heat;

std::cout<<"run 7"<<std::endl;
//print("np_keep",np_keep);
std::cout<<"run 7"<<std::endl;

Fastor::Tensor<float,1,160,480> np_heat_new = np_heat(np_keep)*1;

std::cout<<"run 8"<<std::endl;
//print("np_heat_new",np_heat_new);
std::cout<<"run 8"<<std::endl;

//python code
//np_heat = np_heat.reshape(-1, )#the same as flatten

Fastor::Tensor<float,160*480,1> np_heat_flatten = Fastor::flatten(np_heat_new);

std::cout<<"run 9"<<std::endl;
//print("np_heat_flatten",np_heat_flatten);
std::cout<<"run 9"<<std::endl;

//python code
//np_heat = np.where(np_heat < score_thresh, np.zeros_like(np_heat), np_heat)

Fastor::Tensor<float,160*480,1> score_thresh_matrix;

std::cout<<"run 10"<<std::endl;

score_thresh_matrix.ones();

std::cout<<"run 11"<<std::endl;

score_thresh_matrix=score_thresh_matrix*score_thresh;

std::cout<<"run 12"<<std::endl;
//print("score_thresh_matrix",score_thresh_matrix);
std::cout<<"run 12"<<std::endl;

Fastor::Tensor<bool,160*480,1> np_heat_mask1= score_thresh_matrix<np_heat_flatten;

std::cout<<"run 13"<<std::endl;

//print("np_heat_mask1",np_heat_mask1);
std::cout<<"run 13"<<std::endl;

Fastor::Tensor<float,160*480,1> np_heat_after_choice=np_heat_flatten(np_heat_mask1)*1;


std::cout<<"run 14"<<std::endl;
//print("np_heat_after_choice",np_heat_after_choice);
std::cout<<"run 14"<<std::endl;


//python code

//np_scores, np_indices = np_topk(np_heat, topk_n)
//np_valid_inx = np.where(np_scores > score_thresh)

// (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
//        17, 18, 19], dtype=int64),)

n_topk_res np_scores_with_np_indices =np_topk_160_480(np_heat_after_choice,topk_n);

//std::vector<float> ksd={-2.23,-1.63,-1.36,-1.68,-25.369,-5.954,-0.0};
//std::vector<int> sort_indexs=sort_index(ksd);
//show_vector(sort_indexs);


std::cout<<"run 15"<<std::endl;
std::vector<float> data_sort=np_scores_with_np_indices.topk_data_sort;
std::vector<int> index_sort=np_scores_with_np_indices.top_index_sort;

//show_vector(data_sort);

std::cout<<"\n";

//show_vector(index_sort);



std::cout<<"run 15"<<std::endl;

const size_t top_c = (size_t)topk_n;  
std::cout<<"run 16"<<std::endl;

Fastor::Tensor<float,150,1> np_scores(np_scores_with_np_indices.topk_data_sort);
std::cout<<"run 17"<<std::endl;
//print("np_scores",np_scores);
std::cout<<"run 17"<<std::endl;


Fastor::Tensor<int,150,1> np_indices(np_scores_with_np_indices.top_index_sort);
std::cout<<"run 18"<<std::endl;

Fastor::Tensor<bool,150,1> np_scores_after_mask = np_scores>score_thresh;
std::cout<<"run 19"<<std::endl;


//get the 1 or true 位置

std::vector<int> np_valid_inx_vector;
std::cout<<"run 20"<<std::endl;


for(int index_valid=0;index_valid<150;index_valid++)
{
    if(np_scores_after_mask(index_valid,0)==1)
    {
        np_valid_inx_vector.push_back(index_valid);
    }
}

std::cout<<"run 21"<<std::endl;
//show_vector(np_valid_inx_vector);
std::cout<<"run 21"<<std::endl;

//python code
//   np_scores = np_scores[np_valid_inx]
//     np_indices = np_indices[np_valid_inx]
//     print("\n")
//     #print(np_scores)

//     print("\n")
//     #print(np_indices)
//     #simple


std::vector<float> np_scores_after_valid;
std::vector<int> np_indices_after_valid;
std::cout<<"run 22"<<std::endl;

for(int index_valid_=0;index_valid_<np_valid_inx_vector.size();index_valid_++)
{
    np_scores_after_valid.push_back(np_scores(index_valid_,0));
    np_indices_after_valid.push_back(np_indices(index_valid_,0));
}
std::cout<<"run 23"<<std::endl;

//show_vector(np_scores_after_valid);

//show_vector(np_indices_after_valid);



//python code
//     np_yy = np.floor_divide(np_indices, np_w)[:, np.newaxis]#every element divide add a dim
//     print("\n")
//     print(np_yy)


//     np_xx = np.fmod(np_indices, np_w)[:, np.newaxis]#every element % operator add a dim
//     print("\n")
//     print(np_xx)


std::vector<float> np_xx;
std::vector<float> np_yy;

std::cout<<"run 24 "<<np_indices_after_valid.size()<<std::endl;

for(int index_np_indices=0;index_np_indices<np_indices_after_valid.size();index_np_indices++)
{
    std::cout<<" the half left "<<np_indices_after_valid[index_np_indices]<<std::endl;
    np_yy.push_back(float(np_indices_after_valid[index_np_indices])/float(np_w));
    np_xx.push_back(float(int(np_indices_after_valid[index_np_indices]%int(np_w))));
}

std::cout<<"run 25"<<std::endl;

show_vector(np_yy);

for(int yy_index=0;yy_index<np_yy.size();yy_index++)
{
    np_yy[yy_index]=(int)(np_yy[yy_index]);
}


show_vector(np_yy);

printf("\n");

show_vector(np_xx);

std::cout<<"run 25"<<std::endl;

Fastor::Tensor<float,20,1> np_xx_tensor(np_xx);
Fastor::Tensor<float,20,1> np_yy_tensor(np_yy);

std::cout<<"run 26"<<std::endl;




//     np_center_ptss = np.concatenate((np_xx, np_yy), axis=-1)

//     print("\n")
//     print(np_center_ptss) (20,2)

//     #depends on column 
Fastor::Tensor<float,20,2> np_center_ptss;
np_center_ptss.ones();

std::cout<<"run 27"<<std::endl;

np_center_ptss(Fastor::all,Fastor::seq(0,1))=np_xx_tensor;
np_center_ptss(Fastor::all,Fastor::seq(1,2))=np_yy_tensor;


print("np_center_ptss",np_center_ptss);


std::cout<<"run 28"<<std::endl;





//     np_start_point = np_center_ptss + np.squeeze(np_displacement[0, :2, np_yy, np_xx])//(20, 1, 2)
//     print("\n")
//     print(np_start_point)


Fastor::Tensor<float,20,2> np_displacement_temp_s;
np_displacement_temp_s.ones();

std::cout<<"run 29"<<std::endl;


for(int index_dim1=0;index_dim1<2;index_dim1++)
{
for(int index_np=0;index_np<20;index_np++)
{
    np_displacement_temp_s(index_np,index_dim1)=np_displacement(0,index_dim1,int(np_yy[index_np]),int(np_xx[index_np]));
}
}

std::cout<<"run 30"<<std::endl;



Fastor::Tensor<float,20,2> np_start_point = np_center_ptss+np_displacement_temp_s;

std::cout<<"run 31"<<std::endl;
//print("np_start_point",np_start_point);
std::cout<<"run 31"<<std::endl;

    
//     np_end_point = np_center_ptss + np.squeeze(np_displacement[0, 2:, np_yy, np_xx])
//     print("\n")
//     print(np_end_point)



Fastor::Tensor<float,20,2> np_displacement_temp_e;
np_displacement_temp_e.ones();

std::cout<<"run 32"<<std::endl;


for(int index_dim2=2;index_dim2<4;index_dim2++)
{
    for(int index_np2=0;index_np2<20;index_np2++)
    {
        np_displacement_temp_e(index_np2,index_dim2-2)=np_displacement(0,index_dim2,int(np_yy[index_np2]),int(np_xx[index_np2]));
    }
}

std::cout<<"run 33"<<std::endl;


Fastor::Tensor<float,20,2> np_end_point=np_center_ptss+np_displacement_temp_e;

std::cout<<"run 34"<<std::endl;
print("np_end_point",np_end_point);
std::cout<<"run 34"<<std::endl;


//     np_lines = np.concatenate((np_start_point, np_end_point), axis=-1)
//     print("\n")
//     print(np_lines)
       
//(20,2 20,2)

Fastor::Tensor<float,20,4> np_lines;
np_lines.ones();

std::cout<<"run 35"<<std::endl;

np_lines(Fastor::all,Fastor::seq(0,2))=np_start_point;
np_lines(Fastor::all,Fastor::seq(2,4))=np_end_point;

std::cout<<"run 36"<<std::endl;
//print("np_lines",np_lines);
std::cout<<"run 36"<<std::endl;




//     np_all_lens = (np_end_point - np_start_point) ** 2
//     print("\n")
//     print(np_all_lens)
Fastor::Tensor<float,20,2> np_all_lens = np_end_point-np_start_point;
std::cout<<"run 37"<<std::endl;


for(int index_dim_c=0;index_dim_c<20;index_dim_c++)
{
    for(int index_dim_r=0;index_dim_r<2;index_dim_r++)
    {
        np_all_lens(index_dim_c,index_dim_r)=np_all_lens(index_dim_c,index_dim_r)*np_all_lens(index_dim_c,index_dim_r);
    }
}

std::cout<<"run 38"<<std::endl;
//print("np_all_lens",np_all_lens);
std::cout<<"run 38"<<std::endl;

//     np_all_lens = np_all_lens.sum(axis=-1)# rows add
//     print("\n")
//     print(np_all_lens) (20,1)

//

Fastor::Tensor<float,20,1> np_all_len_temp;
np_all_len_temp.zeros();
std::cout<<"run 39"<<std::endl;


for(int index_colums=0;index_colums<2;index_colums++)
{
    np_all_len_temp(Fastor::all,Fastor::seq(0,1)) += np_all_lens(Fastor::all,Fastor::seq(index_colums,index_colums+1));
}
std::cout<<"run 40"<<std::endl;

//print("np_all_len_temp",np_all_len_temp);

std::cout<<"run 40"<<std::endl;



//     np_all_lens = np.sqrt(np_all_lens)# every element sqrt
//     print("\n")
//     print(np_all_lens) 
// (20,1)


Fastor::Tensor<float,20,1> np_all_lens_temp_sqrt;
np_all_lens_temp_sqrt.ones();

std::cout<<"run 41"<<std::endl;

for(int index_rows_temp=0;index_rows_temp<20;index_rows_temp++)
{
    np_all_lens_temp_sqrt(index_rows_temp,0)=std:: sqrt(np_all_len_temp(index_rows_temp,0));
}

std::cout<<"run 42"<<std::endl;
//print("np_all_lens_temp_sqrt",np_all_lens_temp_sqrt);

std::cout<<"run 42"<<std::endl;




//     np_valid_inx = np.where(np_all_lens > len_thresh)
//     print("\n")
//     print("\n")
//     print(np_valid_inx)
Fastor::Tensor<float,20,1> len_thresh_matrix;
Fastor::Tensor<float,20,1> len_mask_temp=len_thresh_matrix*len_thresh;
Fastor::Tensor<bool,20,1> np_all_lens_m = np_all_lens_temp_sqrt>len_mask_temp;
std::cout<<"run 43"<<std::endl;


std::vector<int> np_valid_inx;

for(int index_np_all_lens_m=0;index_np_all_lens_m<20;index_np_all_lens_m++)
{
    if(np_all_lens_m(index_np_all_lens_m,0)==1)
    {
        np_valid_inx.push_back(index_np_all_lens_m);
    }
}

std::cout<<"run 44"<<std::endl;
show_vector(np_valid_inx);

std::cout<<"run 44"<<std::endl;


//     np_center_ptss = np_center_ptss[np_valid_inx]
//     print("\n")
//     print(np_center_ptss) (20,2)


std::vector<std::vector<float>> np_center_ptss_v;
for(int row_np_center_ptss=0;row_np_center_ptss<np_valid_inx.size();row_np_center_ptss++)
{
std::vector<float> rows_all;
for(int colum_np_center_ptss=0;colum_np_center_ptss<2;colum_np_center_ptss++)
{
rows_all.push_back(np_center_ptss(row_np_center_ptss,colum_np_center_ptss));
}
np_center_ptss_v.push_back(rows_all);
}
std::cout<<"run 45"<<std::endl;

for(int i_=0;i_<np_center_ptss_v.size();i_++)
{
for(int j_=0;j_<np_center_ptss_v[i_].size();j_++)
{
    std::cout<<np_center_ptss_v[i_][j_]<<" ";
}
std::cout<<"\n";
}

std::cout<<"run 45"<<std::endl;


//     np_lines = np_lines[np_valid_inx] (20,4)
//     print("\n")
//     print(np_lines)
std::vector<std::vector<float>> np_lines_v;
for(int row_np_lines_v=0;row_np_lines_v<np_valid_inx.size();row_np_lines_v++)
{
    std::vector<float> rows_all_t;
    for(int colum_np_lines_v=0;colum_np_lines_v<4;colum_np_lines_v++)
    {
        rows_all_t.push_back(np_lines(row_np_lines_v,colum_np_lines_v));
    }
    np_lines_v.push_back(rows_all_t);
}

std::cout<<"run 46"<<std::endl;
for(int i_=0;i_<np_lines_v.size();i_++)
{
for(int j_=0;j_<np_lines_v[i_].size();j_++)
{
    std::cout<<np_lines_v[i_][j_]<<" ";
}
std::cout<<"\n";
}
std::cout<<"run 46"<<std::endl;



    
//     np_scores = np_scores[np_valid_inx] (20,1)
//     print("\n")
//     print(np_scores)

std::vector<float> np_scores_finall;

for(int index_scores=0;index_scores<np_valid_inx.size();index_scores++)
{
    np_scores_finall.push_back(np_scores_after_valid[index_scores]);
}

std::cout<<"run 47"<<std::endl;
show_vector(np_scores_finall);
std::cout<<"run 47"<<std::endl;

decode_lines_data temp_re;
temp_re.np_scores_finall=np_scores_finall;
temp_re.np_lines_v=np_lines_v;
temp_re.np_center_ptss_v=np_center_ptss_v;

return temp_re;

}




decode_lines_data deccode_lines_48_144(Fastor::Tensor<float,1,9,48,144> np_tpMap, float score_thresh=0.1, float len_thresh=2.0, float topk_n=100.0, float ksize=3.0)
{
//(1, 4, 48, 144)
//np_b, np_c, np_h, np_w = np_tpMap.shape# (1, 9, 48, 144)

int np_b = 1;
int np_c = 4;
int np_h = 48;
int np_w = 144;

Fastor::Tensor<float,1,4,48,144> np_displacement = np_tpMap(Fastor::all, Fastor::seq(1,5), Fastor::all, Fastor::all);

std::cout<<"run 1"<<std::endl;

//print("np_displacement",np_displacement);


std::cout<<"run 1"<<std::endl;

//np_center = np_tpMap[:, 0, :, :]
Fastor::Tensor<float,1,48,144> np_center = np_tpMap(Fastor::all,0,Fastor::all,Fastor::all);


std::cout<<"run 2"<<std::endl;
//print("np_center",np_center);
std::cout<<"run 2"<<std::endl;

Fastor::Tensor<float,1,48,144> np_center_oppo = -np_center;

std::cout<<"run 3"<<std::endl;
//print("np_center_oppo",np_center_oppo);
std::cout<<"run 3"<<std::endl;

// np_heat = 1 / (1 + np.exp(-np_center))

Fastor::Tensor<float,1,48,144> np_heat = 1.0/(1.0+exp(np_center_oppo));

std::cout<<"run 4"<<std::endl;
//print("np_heat",np_heat);

std::cout<<"run 4"<<std::endl;

std::vector<int> ksize_={int(ksize),int(ksize)};

std::cout<<"run 5"<<std::endl;

Fastor::Tensor<float,1,48,144> np_hmax = np_max_pool2d_48_144(np_heat, ksize_, 1, 1);



std::cout<<"run 6"<<std::endl;
//print("np_hmax",np_hmax);
std::cout<<"run 6"<<std::endl;

//python code
//np_keep = (np_hmax == np_heat).astype('float32')
Fastor::Tensor<bool,1,48,144> np_keep = np_hmax==np_heat;

std::cout<<"run 7"<<std::endl;
//print("np_keep",np_keep);
std::cout<<"run 7"<<std::endl;

Fastor::Tensor<float,1,48,144> np_heat_new = np_heat(np_keep)*1;

std::cout<<"run 8"<<std::endl;
//print("np_heat_new",np_heat_new);
std::cout<<"run 8"<<std::endl;

//python code
//np_heat = np_heat.reshape(-1, )#the same as flatten

Fastor::Tensor<float,48*144,1> np_heat_flatten = Fastor::flatten(np_heat_new);

std::cout<<"run 9"<<std::endl;
//print("np_heat_flatten",np_heat_flatten);
std::cout<<"run 9"<<std::endl;

//python code
//np_heat = np.where(np_heat < score_thresh, np.zeros_like(np_heat), np_heat)

Fastor::Tensor<float,48*144,1> score_thresh_matrix;

std::cout<<"run 10"<<std::endl;

score_thresh_matrix.ones();

std::cout<<"run 11"<<std::endl;

score_thresh_matrix=score_thresh_matrix*score_thresh;

std::cout<<"run 12"<<std::endl;
//print("score_thresh_matrix",score_thresh_matrix);
std::cout<<"run 12"<<std::endl;

Fastor::Tensor<bool,48*144,1> np_heat_mask1= score_thresh_matrix<np_heat_flatten;

std::cout<<"run 13"<<std::endl;

//print("np_heat_mask1",np_heat_mask1);
std::cout<<"run 13"<<std::endl;

Fastor::Tensor<float,48*144,1> np_heat_after_choice=np_heat_flatten(np_heat_mask1)*1;


std::cout<<"run 14"<<std::endl;
//print("np_heat_after_choice",np_heat_after_choice);
std::cout<<"run 14"<<std::endl;


//python code

//np_scores, np_indices = np_topk(np_heat, topk_n)
//np_valid_inx = np.where(np_scores > score_thresh)

// (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
//        17, 18, 19], dtype=int64),)

n_topk_res np_scores_with_np_indices =np_topk_48_144(np_heat_after_choice,topk_n);

//std::vector<float> ksd={-2.23,-1.63,-1.36,-1.68,-25.369,-5.954,-0.0};
//std::vector<int> sort_indexs=sort_index(ksd);
//show_vector(sort_indexs);


std::cout<<"run 15"<<std::endl;
std::vector<float> data_sort=np_scores_with_np_indices.topk_data_sort;
std::vector<int> index_sort=np_scores_with_np_indices.top_index_sort;

//show_vector(data_sort);

std::cout<<"\n";

//show_vector(index_sort);



std::cout<<"run 15"<<std::endl;

const size_t top_c = (size_t)topk_n;  
std::cout<<"run 16"<<std::endl;

Fastor::Tensor<float,150,1> np_scores(np_scores_with_np_indices.topk_data_sort);
std::cout<<"run 17"<<std::endl;
//print("np_scores",np_scores);
std::cout<<"run 17"<<std::endl;


Fastor::Tensor<int,150,1> np_indices(np_scores_with_np_indices.top_index_sort);
std::cout<<"run 18"<<std::endl;

Fastor::Tensor<bool,150,1> np_scores_after_mask = np_scores>score_thresh;
std::cout<<"run 19"<<std::endl;


//get the 1 or true 位置

std::vector<int> np_valid_inx_vector;
std::cout<<"run 20"<<std::endl;


for(int index_valid=0;index_valid<150;index_valid++)
{
    if(np_scores_after_mask(index_valid,0)==1)
    {
        np_valid_inx_vector.push_back(index_valid);
    }
}

std::cout<<"run 21"<<std::endl;
//show_vector(np_valid_inx_vector);
std::cout<<"run 21"<<std::endl;

//python code
//   np_scores = np_scores[np_valid_inx]
//     np_indices = np_indices[np_valid_inx]
//     print("\n")
//     #print(np_scores)

//     print("\n")
//     #print(np_indices)
//     #simple


std::vector<float> np_scores_after_valid;
std::vector<int> np_indices_after_valid;
std::cout<<"run 22"<<std::endl;

for(int index_valid_=0;index_valid_<np_valid_inx_vector.size();index_valid_++)
{
    np_scores_after_valid.push_back(np_scores(index_valid_,0));
    np_indices_after_valid.push_back(np_indices(index_valid_,0));
}
std::cout<<"run 23"<<std::endl;

//show_vector(np_scores_after_valid);

//show_vector(np_indices_after_valid);



//python code
//     np_yy = np.floor_divide(np_indices, np_w)[:, np.newaxis]#every element divide add a dim
//     print("\n")
//     print(np_yy)


//     np_xx = np.fmod(np_indices, np_w)[:, np.newaxis]#every element % operator add a dim
//     print("\n")
//     print(np_xx)


std::vector<float> np_xx;
std::vector<float> np_yy;

std::cout<<"run 24 "<<np_indices_after_valid.size()<<std::endl;

for(int index_np_indices=0;index_np_indices<np_indices_after_valid.size();index_np_indices++)
{
    std::cout<<" the half left "<<np_indices_after_valid[index_np_indices]<<std::endl;
    np_yy.push_back(float(np_indices_after_valid[index_np_indices])/float(np_w));
    np_xx.push_back(float(int(np_indices_after_valid[index_np_indices]%int(np_w))));
}

std::cout<<"run 25"<<std::endl;

show_vector(np_yy);

for(int yy_index=0;yy_index<np_yy.size();yy_index++)
{
    np_yy[yy_index]=(int)(np_yy[yy_index]);
}


show_vector(np_yy);

printf("\n");

show_vector(np_xx);

std::cout<<"run 25"<<std::endl;

Fastor::Tensor<float,20,1> np_xx_tensor(np_xx);
Fastor::Tensor<float,20,1> np_yy_tensor(np_yy);

std::cout<<"run 26"<<std::endl;




//     np_center_ptss = np.concatenate((np_xx, np_yy), axis=-1)

//     print("\n")
//     print(np_center_ptss) (20,2)

//     #depends on column 
Fastor::Tensor<float,20,2> np_center_ptss;
np_center_ptss.ones();

std::cout<<"run 27"<<std::endl;

np_center_ptss(Fastor::all,Fastor::seq(0,1))=np_xx_tensor;
np_center_ptss(Fastor::all,Fastor::seq(1,2))=np_yy_tensor;


print("np_center_ptss",np_center_ptss);


std::cout<<"run 28"<<std::endl;





//     np_start_point = np_center_ptss + np.squeeze(np_displacement[0, :2, np_yy, np_xx])//(20, 1, 2)
//     print("\n")
//     print(np_start_point)


Fastor::Tensor<float,20,2> np_displacement_temp_s;
np_displacement_temp_s.ones();

std::cout<<"run 29"<<std::endl;


for(int index_dim1=0;index_dim1<2;index_dim1++)
{
for(int index_np=0;index_np<20;index_np++)
{
    np_displacement_temp_s(index_np,index_dim1)=np_displacement(0,index_dim1,int(np_yy[index_np]),int(np_xx[index_np]));
}
}

std::cout<<"run 30"<<std::endl;



Fastor::Tensor<float,20,2> np_start_point = np_center_ptss+np_displacement_temp_s;

std::cout<<"run 31"<<std::endl;
//print("np_start_point",np_start_point);
std::cout<<"run 31"<<std::endl;

    
//     np_end_point = np_center_ptss + np.squeeze(np_displacement[0, 2:, np_yy, np_xx])
//     print("\n")
//     print(np_end_point)



Fastor::Tensor<float,20,2> np_displacement_temp_e;
np_displacement_temp_e.ones();

std::cout<<"run 32"<<std::endl;


for(int index_dim2=2;index_dim2<4;index_dim2++)
{
    for(int index_np2=0;index_np2<20;index_np2++)
    {
        np_displacement_temp_e(index_np2,index_dim2-2)=np_displacement(0,index_dim2,int(np_yy[index_np2]),int(np_xx[index_np2]));
    }
}

std::cout<<"run 33"<<std::endl;


Fastor::Tensor<float,20,2> np_end_point=np_center_ptss+np_displacement_temp_e;

std::cout<<"run 34"<<std::endl;
print("np_end_point",np_end_point);
std::cout<<"run 34"<<std::endl;


//     np_lines = np.concatenate((np_start_point, np_end_point), axis=-1)
//     print("\n")
//     print(np_lines)
       
//(20,2 20,2)

Fastor::Tensor<float,20,4> np_lines;
np_lines.ones();

std::cout<<"run 35"<<std::endl;

np_lines(Fastor::all,Fastor::seq(0,2))=np_start_point;
np_lines(Fastor::all,Fastor::seq(2,4))=np_end_point;

std::cout<<"run 36"<<std::endl;
//print("np_lines",np_lines);
std::cout<<"run 36"<<std::endl;




//     np_all_lens = (np_end_point - np_start_point) ** 2
//     print("\n")
//     print(np_all_lens)
Fastor::Tensor<float,20,2> np_all_lens = np_end_point-np_start_point;
std::cout<<"run 37"<<std::endl;


for(int index_dim_c=0;index_dim_c<20;index_dim_c++)
{
    for(int index_dim_r=0;index_dim_r<2;index_dim_r++)
    {
        np_all_lens(index_dim_c,index_dim_r)=np_all_lens(index_dim_c,index_dim_r)*np_all_lens(index_dim_c,index_dim_r);
    }
}

std::cout<<"run 38"<<std::endl;
//print("np_all_lens",np_all_lens);
std::cout<<"run 38"<<std::endl;

//     np_all_lens = np_all_lens.sum(axis=-1)# rows add
//     print("\n")
//     print(np_all_lens) (20,1)

//

Fastor::Tensor<float,20,1> np_all_len_temp;
np_all_len_temp.zeros();
std::cout<<"run 39"<<std::endl;


for(int index_colums=0;index_colums<2;index_colums++)
{
    np_all_len_temp(Fastor::all,Fastor::seq(0,1)) += np_all_lens(Fastor::all,Fastor::seq(index_colums,index_colums+1));
}
std::cout<<"run 40"<<std::endl;

//print("np_all_len_temp",np_all_len_temp);

std::cout<<"run 40"<<std::endl;



//     np_all_lens = np.sqrt(np_all_lens)# every element sqrt
//     print("\n")
//     print(np_all_lens) 
// (20,1)


Fastor::Tensor<float,20,1> np_all_lens_temp_sqrt;
np_all_lens_temp_sqrt.ones();

std::cout<<"run 41"<<std::endl;

for(int index_rows_temp=0;index_rows_temp<20;index_rows_temp++)
{
    np_all_lens_temp_sqrt(index_rows_temp,0)=std:: sqrt(np_all_len_temp(index_rows_temp,0));
}

std::cout<<"run 42"<<std::endl;
//print("np_all_lens_temp_sqrt",np_all_lens_temp_sqrt);

std::cout<<"run 42"<<std::endl;




//     np_valid_inx = np.where(np_all_lens > len_thresh)
//     print("\n")
//     print("\n")
//     print(np_valid_inx)
Fastor::Tensor<float,20,1> len_thresh_matrix;
Fastor::Tensor<float,20,1> len_mask_temp=len_thresh_matrix*len_thresh;
Fastor::Tensor<bool,20,1> np_all_lens_m = np_all_lens_temp_sqrt>len_mask_temp;
std::cout<<"run 43"<<std::endl;


std::vector<int> np_valid_inx;

for(int index_np_all_lens_m=0;index_np_all_lens_m<20;index_np_all_lens_m++)
{
    if(np_all_lens_m(index_np_all_lens_m,0)==1)
    {
        np_valid_inx.push_back(index_np_all_lens_m);
    }
}

std::cout<<"run 44"<<std::endl;
show_vector(np_valid_inx);

std::cout<<"run 44"<<std::endl;


//     np_center_ptss = np_center_ptss[np_valid_inx]
//     print("\n")
//     print(np_center_ptss) (20,2)


std::vector<std::vector<float>> np_center_ptss_v;
for(int row_np_center_ptss=0;row_np_center_ptss<np_valid_inx.size();row_np_center_ptss++)
{
std::vector<float> rows_all;
for(int colum_np_center_ptss=0;colum_np_center_ptss<2;colum_np_center_ptss++)
{
rows_all.push_back(np_center_ptss(row_np_center_ptss,colum_np_center_ptss));
}
np_center_ptss_v.push_back(rows_all);
}
std::cout<<"run 45"<<std::endl;

for(int i_=0;i_<np_center_ptss_v.size();i_++)
{
for(int j_=0;j_<np_center_ptss_v[i_].size();j_++)
{
    std::cout<<np_center_ptss_v[i_][j_]<<" ";
}
std::cout<<"\n";
}

std::cout<<"run 45"<<std::endl;


//     np_lines = np_lines[np_valid_inx] (20,4)
//     print("\n")
//     print(np_lines)
std::vector<std::vector<float>> np_lines_v;
for(int row_np_lines_v=0;row_np_lines_v<np_valid_inx.size();row_np_lines_v++)
{
    std::vector<float> rows_all_t;
    for(int colum_np_lines_v=0;colum_np_lines_v<4;colum_np_lines_v++)
    {
        rows_all_t.push_back(np_lines(row_np_lines_v,colum_np_lines_v));
    }
    np_lines_v.push_back(rows_all_t);
}

std::cout<<"run 46"<<std::endl;
for(int i_=0;i_<np_lines_v.size();i_++)
{
for(int j_=0;j_<np_lines_v[i_].size();j_++)
{
    std::cout<<np_lines_v[i_][j_]<<" ";
}
std::cout<<"\n";
}
std::cout<<"run 46"<<std::endl;



    
//     np_scores = np_scores[np_valid_inx] (20,1)
//     print("\n")
//     print(np_scores)

std::vector<float> np_scores_finall;

for(int index_scores=0;index_scores<np_valid_inx.size();index_scores++)
{
    np_scores_finall.push_back(np_scores_after_valid[index_scores]);
}

std::cout<<"run 47"<<std::endl;
show_vector(np_scores_finall);
std::cout<<"run 47"<<std::endl;

decode_lines_data temp_re;
temp_re.np_scores_finall=np_scores_finall;
temp_re.np_lines_v=np_lines_v;
temp_re.np_center_ptss_v=np_center_ptss_v;

return temp_re;

}




decode_lines_data mlsd_post_process_16_160_480(float *arr2_data)
{
    //

    //there start
    Fastor::Tensor<float,1,16,160,480> tn1(arr2_data);


    

    Fastor::Tensor<float,1,16,160,480> batch_outputs = tn1(0,Fastor::all,Fastor::all,Fastor::all);

    

    Fastor::Tensor<float,1,9,160,480> tp_mask = batch_outputs(Fastor::all,Fastor::seq(7,16),Fastor::all,Fastor::all);

    

    //std::cout<<"tp_mask size: "<<tp_mask.dimension(2)<<std::endl;
    

    decode_lines_data temp_n=deccode_lines_160_480(tp_mask, 0.2, 5.0, 150, 3.0);

    //
    return temp_n;
}


decode_lines_data mlsd_post_process_16_48_144(float *arr2_data)
{
    //

    //there start
    Fastor::Tensor<float,1,16,48,144> tn1(arr2_data);


    

    Fastor::Tensor<float,1,16,48,144> batch_outputs = tn1(0,Fastor::all,Fastor::all,Fastor::all);

    

    Fastor::Tensor<float,1,9,48,144> tp_mask = batch_outputs(Fastor::all,Fastor::seq(7,16),Fastor::all,Fastor::all);

    

    //std::cout<<"tp_mask size: "<<tp_mask.dimension(2)<<std::endl;
    

    decode_lines_data temp_n=deccode_lines_48_144(tp_mask, 0.2, 5.0, 150, 3.0);

    //
    return temp_n;
}


decode_lines_data mlsd_post_process_16_80_240(float *arr2_data)
{
    //there start
    Fastor::Tensor<float,1,16,80,240> tn1(arr2_data);


    

    Fastor::Tensor<float,1,16,80,240> batch_outputs = tn1(0,Fastor::all,Fastor::all,Fastor::all);

    

    Fastor::Tensor<float,1,9,80,240> tp_mask = batch_outputs(Fastor::all,Fastor::seq(7,16),Fastor::all,Fastor::all);

    

    //std::cout<<"tp_mask size: "<<tp_mask.dimension(2)<<std::endl;
    

    decode_lines_data temp_n=deccode_lines_80_240(tp_mask, 0.2, 5.0, 150, 3.0);
}




xt::xtensor<double,3> np_max_pool2d_xtensor(xt::xtensor<double,3> mat, std::vector<int> kernel, int stride, int padding)
{

    int mat_c = mat.shape(0);
    int mat_h = mat.shape(1);
    int mat_w = mat.shape(2);
    
    xt::xtensor<double,3> mat_padding = xt::pad(mat,{{0,0},{padding,padding},{padding,padding}},xt::pad_mode::constant,0);

    std::cout<<"mat_padding"<<std::endl;
    std::cout<<mat_padding<<std::endl;


    int new_h = floor(float(mat_h + 2 * padding - (kernel[1] - 1) - 1) / float(stride)) + 1;
    int new_w = floor(float(mat_w + 2 * padding - (kernel[0] - 1) - 1) / float(stride)) + 1;

    printf("new_h:%d\n",new_h);
    printf("new_w:%d\n",new_w);

    // python result = np.zeros((mat_c, new_h, new_w))

    xt::xtensor<double,3> result = xt::zeros<double>({mat_c,new_h,new_w});

    std::cout<<"result"<<std::endl;
    std::cout<<result<<std::endl;

    // python code
    // for c_idx in range(mat_c):
    //     #print("c_idx ",c_idx)
    //     for h_idx in range(0, new_h, stride):
    //         for w_idx in range(0, new_w, stride):
    //             temp = np.max(mat[c_idx][h_idx:h_idx + kernel[0], w_idx:w_idx + kernel[1]])#某个维度上的最大值
    //             #切片
    //             result[c_idx][h_idx][w_idx] = temp
    // return result
    for(int c_idx=0;c_idx<mat_c;c_idx++)
    {
        for(int h_idx=0;h_idx<new_h;h_idx+=stride)
        {
            for(int w_idx=0;w_idx<new_w;w_idx+=stride)
            {
                double temp=xt::amax(xt::view(mat_padding,c_idx,xt::range(h_idx,h_idx+kernel[0]),xt::range(w_idx,w_idx+kernel[1])))(0);
                //std::cout<<temp<<std::endl;
                result(c_idx,h_idx,w_idx)=temp;
            }
        }
    }

    std::cout<<"result"<<std::endl;
    //std::cout<<result<<std::endl;
    return result;
}


struct np_topk_res
{
xt::xtensor<double,2> topk_index_sort;
xt::xtensor<double,2> topk_data_sort;

};



//python code
np_topk_res np_topk(xt::xtensor<double,2> mat, int topk_n)
{
    //print("\n")
    //print("###########in np_topk#########")
    

    //print("in np_topk")
    //print(mat)
    //print("in np_topk")
    
    //topk_data_sort = -np.sort(-mat)[:topk_n]#sort on rows

    xt::xtensor<double,2> mat_opposite = -mat;
    std::cout<<"mat_opposite"<<std::endl;
    std::cout<<mat_opposite<<std::endl;


    xt::xtensor<double,2> mat_opposite_sort=xt::sort(mat_opposite,0);
    std::cout<<"mat_opposite_sort"<<std::endl;
    std::cout<<mat_opposite_sort<<std::endl;

    xt::xtensor<double,2> topk_data_sort=xt::view(xt::sort(mat_opposite,0),xt::range(0,topk_n),xt::all());
    std::cout<<"topk_data_sort"<<std::endl;
    topk_data_sort=-topk_data_sort;
    std::cout<<topk_data_sort<<std::endl;


    // topk_index_sort = np.argsort(-mat)[:topk_n]#sort return position np.array([3, 1, 2]) #[1 2 0]

    xt::xtensor<double,2> topk_index_sort = xt::view(xt::argsort(mat_opposite,0),xt::range(0,topk_n),xt::all());

    std::cout<<"topk_index_sort"<<std::endl;
    std::cout<<topk_index_sort<<std::endl;
    
    // print(topk_data_sort.shape)
    // print(topk_index_sort.shape)

    // print("###########in np_topk#########")

    np_topk_res temp_r;
    temp_r.topk_index_sort=topk_index_sort;
    temp_r.topk_data_sort=topk_data_sort;
    
    // return topk_data_sort, topk_index_sort
    return temp_r;
}


struct lines_all
{

xt::xtensor<double,2> np_center_ptss;
xt::xtensor<double,2> np_lines;
xt::xtensor<double,2> np_scores;

};

struct lines_single
{
float start_pointx;
float start_pointy;
float end_pointx;
float end_pointy;
float scores;
};




std::vector<lines_single>  deccode_lines_xtensor(xt::xtensor<double,4> np_tpMap,float score_thresh=0.1, int len_thresh=2, int topk_n=100, int ksize=3)
{
    std::cout<<np_tpMap<<std::endl;
    int np_b = np_tpMap.shape(0);
    int np_c = np_tpMap.shape(1);
    int np_h = np_tpMap.shape(2);
    int np_w = np_tpMap.shape(3);
    std::cout<<"shape :"<<np_b<<" "<<np_c<<" "<<np_h<<" "<<np_w<<std::endl;
    xt::xtensor<double,4> np_displacement = xt::view(np_tpMap,xt::all(),xt::range(1,5),xt::all(),xt::all());

    //python code np_center = np_tpMap[:, 0, :, :]

    xt::xtensor<double,3> np_center = xt::view(np_tpMap,xt::all(),0,xt::all(),xt::all());
    std::cout<<"shape np_center : "<<np_center.shape(0)<<" "<<np_center.shape(1)<<" "<<np_center.shape(2)<<std::endl;
    std::cout<<np_center<<std::endl;

    //python code np_heat = 1 / (1 + np.exp(-np_center))
    xt::xtensor<double,3> np_heat = 1/(1+xt::exp(-np_center));

    std::cout<<"np_heat"<<std::endl;
    std::cout<<np_heat<<std::endl;
    xt::xtensor<double,3> np_hmax =np_max_pool2d_xtensor(np_heat,{ksize,ksize},1,int(floor(float(ksize - 1)/2.0)));
    std::cout<<"np_hmax"<<std::endl;
    std::cout<<np_hmax<<std::endl;

    //python code np_keep = (np_hmax == np_heat).astype('float32')

    xt::xtensor<double,3> np_keep = xt::cast<double>(xt::equal(np_hmax,np_heat));

    std::cout<<"np_keep"<<np_keep<<std::endl;

    //python code np_heat = np_heat * np_keep
    xt::xtensor<double,3> np_heat_new =np_heat*np_keep;

    std::cout<<"np_keep "<<np_heat_new<<std::endl;
    std::cout<<"np_keep shape :"<<np_heat_new.shape(0)<<" "<<np_heat_new.shape(1)<<" "<<np_heat_new.shape(2)<<std::endl;

    //python code np_heat = np_heat.reshape(-1, )

    xt::xtensor<double,3> np_heat_dim1=np_heat_new.reshape({1,-1,1});

    std::cout<<"np_heat_dim1"<<std::endl;
    std::cout<<np_heat_dim1<<std::endl;

    xt::xtensor<double,2> np_heat_2dim=xt::view(np_heat_dim1,0,xt::all(),xt::all());

    std::cout<<"np_heat_2dim"<<std::endl;
    std::cout<<np_heat_2dim<<std::endl;

    

    xt::xtensor<double,2> temp_np_heat=xt::zeros<double>({np_heat_2dim.shape(0),np_heat_2dim.shape(1)});

    xt::xtensor<double,2> np_heat_after_where=xt::where(np_heat_2dim < score_thresh, temp_np_heat, np_heat_2dim);
    std::cout<<"np_heat_after_where "<<std::endl;


    np_topk_res temp_res=np_topk(np_heat_after_where,topk_n);


    xt::xtensor<double,2> np_scores =temp_res.topk_data_sort;
    xt::xtensor<double,2> np_indices = temp_res.topk_index_sort;


    //python code np_valid_inx = np.where(np_scores > score_thresh)

    auto np_valid_inx=xt::where(np_scores>score_thresh);

    std::cout<<"np_valid_inx :"<<std::endl;

    std::vector<int> np_valid_inx_;
    std::vector<double> np_scores_afterprocess_value;
    std::vector<double> np_indices_afterprocess_value;


    for(int index_=0;index_<np_valid_inx.size()-1;index_++)
    {
        for(int index_r=0;index_r<np_valid_inx[index_].size();index_r++)
            {

                std::cout<<np_valid_inx[index_][index_r]<<" ";
                np_valid_inx_.push_back(np_valid_inx[index_][index_r]);
                np_scores_afterprocess_value.push_back(np_scores(np_valid_inx[index_][index_r],0));
                np_indices_afterprocess_value.push_back(np_indices(np_valid_inx[index_][index_r],0));
            }
        std::cout<<"\n";
    }



    


    xt::xtensor<double, 2>::shape_type shape_np_scores_afterprocess = {np_valid_inx_.size(),1};
    xt::xtensor<double,2> np_scores_afterprocess = xt::adapt(np_scores_afterprocess_value,shape_np_scores_afterprocess);

    std::cout<<"np_scores_afterprocess"<<std::endl;
    std::cout<<np_scores_afterprocess<<std::endl;

    xt::xtensor<double, 2>::shape_type shape_np_indices_afterprocess = {np_valid_inx_.size(),1};
    xt::xtensor<double,2> np_indices_afterprocess = xt::adapt(np_indices_afterprocess_value,shape_np_indices_afterprocess);
    

    std::cout<<"np_indices_afterprocess"<<std::endl;
    std::cout<<np_indices_afterprocess<<std::endl;




    xt::xtensor<int,2> np_yy_int=xt::cast<double>(np_indices_afterprocess/np_w);
    xt::xtensor<double,2> np_yy = xt::cast<double> (np_yy_int);
    std::cout<<"np_yy"<<std::endl;
    std::cout<<np_yy<<std::endl;


    xt::xtensor<double,2> np_xx=xt::fmod(np_indices_afterprocess,np_w);
    std::cout<<"np_xx"<<std::endl;
    std::cout<<np_xx<<std::endl;


    
    xt::xtensor<double,2> np_center_ptss=xt::concatenate(xtuple(np_xx, np_yy), 1);
    std::cout<<"np_center_ptss"<<std::endl;
    std::cout<<np_center_ptss<<std::endl;


    //python code np_start_point = np_center_ptss + np.squeeze(np_displacement[0, :2, np_yy, np_xx])
    
    //xt::view(np_displacement);

    xt::xarray<int> np_yy_1d= xt::view(xt::cast<int>(np_yy),xt::all(),0);
    std::cout<<"np_yy_1d"<<std::endl;
    std::cout<<np_yy_1d<<std::endl;

    xt::xarray<int> np_xx_1d= xt::view(xt::cast<int>(np_xx),xt::all(),0);
    std::cout<<"np_xx_1d"<<std::endl;
    std::cout<<np_xx_1d<<std::endl;


    //python code np_end_point = np_center_ptss + np.squeeze(np_displacement[0, 2:, np_yy, np_xx])
    //xt::xtensor<double,3> temp_test_range=xt::squeeze(xt::view(np_displacement,0,xt::range(xt::placeholders::_,2),xt::keep(np_yy_1d),xt::keep(np_xx_1d)));
    //std::cout<<"test"<<std::endl;
    //std::cout<<temp_test_range.shape(0)<<" "<<temp_test_range.shape(1)<<" "<<temp_test_range.shape(2)<<std::endl;


    xt::xtensor<double,2> temp_np_start_point = xt::zeros<double>({int(np_xx_1d.shape(0)),2});

    for(int dim1_np_displacement=0;dim1_np_displacement<2;dim1_np_displacement++)
    {
    for(int ins=0;ins<np_xx_1d.shape(0);ins++)
    {   
        temp_np_start_point(ins,dim1_np_displacement)=np_displacement(0,dim1_np_displacement,np_yy_1d(ins),np_xx_1d(ins));
    }
    }

    std::cout<<"temp_np_start_point"<<std::endl;
    std::cout<<temp_np_start_point<<std::endl;

    xt::xtensor<double,2> np_start_point = np_center_ptss+temp_np_start_point;

    std::cout<<"np_start_point"<<std::endl;

    std::cout<<np_start_point<<std::endl;

    xt::xtensor<double,2> temp_np_end_point = xt::zeros<double>({int(np_xx_1d.shape(0)),2});

    for(int dim1_np_displacement=2;dim1_np_displacement<4;dim1_np_displacement++)
    {
        for(int ins=0;ins<np_xx_1d.shape(0);ins++)
        {
            temp_np_end_point(ins,dim1_np_displacement-2)=np_displacement(0,dim1_np_displacement,np_yy_1d(ins),np_xx_1d(ins));
        }
    }

    std::cout<<"temp_np_end_point"<<std::endl;
    std::cout<<temp_np_end_point<<std::endl;


    xt::xtensor<double,2> np_end_point = np_center_ptss+temp_np_end_point;

    std::cout<<"np_end_point"<<std::endl;

    std::cout<<np_end_point<<std::endl;


    xt::xtensor<double,2> np_lines = xt::concatenate(xtuple(np_start_point,np_end_point),1);
    std::cout<<"np_lines"<<std::endl;
    std::cout<<np_lines<<std::endl;


    xt::xtensor<double,2> np_all_lens = xt::pow((np_end_point - np_start_point),2);

    std::cout<<"np_all_lens"<<std::endl;
    std::cout<<np_all_lens<<std::endl;


    xt::xtensor<double,1> sum_np_all_lens=xt::sum(np_all_lens, 1);
    std::cout<<"sum_np_all_lens"<<std::endl;
    std::cout<<sum_np_all_lens<<std::endl;

    xt::xtensor<double,1> sqrt_np_all_lens=xt::sqrt(sum_np_all_lens);
    std::cout<<"sqrt_np_all_lens"<<std::endl;
    std::cout<<sqrt_np_all_lens<<std::endl;


    auto np_valid_inx_index2=xt::where(sqrt_np_all_lens>len_thresh);
    std::vector<int> np_valid_inx_index2_list;

    for(int dim1_a=0;dim1_a<np_valid_inx_index2.size();dim1_a++)
    {
        for(int dim2_a=0;dim2_a<np_valid_inx_index2[dim1_a].size();dim2_a++)
        {
            std::cout<<np_valid_inx_index2[dim1_a][dim2_a]<<" ";
            np_valid_inx_index2_list.push_back(np_valid_inx_index2[dim1_a][dim2_a]);
        }
        std::cout<<"\n";
    }


    xt::xtensor<double,2> np_center_ptss_slice = xt::view(np_center_ptss,xt::keep(np_valid_inx_index2_list),xt::all());
    std::cout<<"np_center_ptss_slice"<<std::endl;
    std::cout<<np_center_ptss_slice<<std::endl;


    xt::xtensor<double,2> np_lines_slice = xt::view(np_lines,xt::keep(np_valid_inx_index2_list),xt::all());
    std::cout<<"np_lines_slice"<<std::endl;
    std::cout<<np_lines_slice<<std::endl;

    xt::xtensor<double,2> np_scores_slice =xt::view(np_scores_afterprocess,xt::keep(np_valid_inx_index2_list),xt::all());
    std::cout<<"np_np_scores_sliceslice"<<std::endl;
    std::cout<<np_scores_slice<<std::endl;

    std::cout<<"np_lines_slice shape 1 :"<<np_lines_slice.shape(0)<<" np_lines_slice shape 2 :"<<np_lines_slice.shape(1)<<std::endl;
    
    std::vector<lines_single> sets_all;
    
    for(int lines_num=0;lines_num<np_lines_slice.shape(0);lines_num++)
    {
        lines_single temp_lines;
        temp_lines.start_pointx=np_lines_slice(lines_num,0);
        temp_lines.start_pointy=np_lines_slice(lines_num,1);
        temp_lines.end_pointx=np_lines_slice(lines_num,2);
        temp_lines.end_pointy=np_lines_slice(lines_num,3);
        temp_lines.scores=np_scores_slice(lines_num,0);
        sets_all.push_back(temp_lines);
    }

    return sets_all;

    //std::cout<<xt::print_options::threshold(10000)<<np_heat_after_where<<std::endl;

}


std::vector<lines_single> mlsd_postprocess_xtensor(int model_dim1,int model_dim2,int model_dim3,float * data)
{
    int dim0=1;
    int dim1=model_dim1;
    int dim2=model_dim2;
    int dim3=model_dim3;

    std::vector<float> results(dim1*dim2*dim3);//注意这里必须要将类型写对否则结果不正确
    
    memcpy( &results[0],data,sizeof(float)*dim1*dim2*dim3);

    // for(int index=0;index<16*48*144;index++)
    // {
    //     results[index]=arr2_data[index];
    // }


    //show_vector(results);

    xt::xtensor<double, 4>::shape_type shape_x = {dim0,dim1,dim2,dim3};
    xt::xtensor<double,4> tn1 = xt::adapt(results,shape_x);

    std::cout<<tn1<<std::endl;


    std::cout<<"#######################"<<std::endl;

    xt::xtensor<double,3> batch_outputs =xt::view(tn1, 0,xt::all(),xt::all(),xt::all());
    std::cout<<batch_outputs<<std::endl;


    

    xt::xtensor<double,4> tp_mask = xt::view(tn1,xt::all(),xt::range(7,16),xt::all(),xt::all());

    std::cout<<"tp_mask"<<tp_mask<<std::endl;

    std::cout<<"tp_mask shape "<<xt::adapt(tp_mask.shape())<<std::endl;


    //SCORE_THRESH = 0.2
    //LENGTH_THRESH = 5
    //TOPK_N = 150
    std::vector<lines_single> result_lines=deccode_lines_xtensor(tp_mask,0.2,5,150,3);
    
    return  result_lines;

}


int main()
{
    //int Nx=16;
    //int Ny=48;
    //int Nz=144;
    //cnpy::npz_t arr = cnpy::npz_load("mlsd.npz");
    //cnpy::NpyArray arr2 = cnpy::npy_load("save_weights.npy");
    //std::cout<<"the arr shape size "<<arr2.shape.size()<<" shape[0] "<<arr2.shape[0]<<" shape[1] "<<arr2.shape[1]<<" shape[2] "<<arr2.shape[2]<<" shape[3] "<<arr2.shape[3]<<std::endl;
    //this will cause crash
    
    cnpy::NpyArray arr3 =cnpy::npy_load("save_weights_origin.npy");
    std::cout<<"the arr shape size3 "<<arr3.shape.size()<<" shape[0] " <<arr3.shape[0]<<" shape[1] "<<arr3.shape[1]<<" shape[3] "<<arr3.shape[2]<<" shape[3] "<<arr3.shape[3]<<" shape[4] "<<arr3.shape[4]<<std::endl;


    int dim0=1;
    int dim1=16;
    int dim2=48;
    int dim3=144;

    //test xtensor
    // std::vector<float> results(dim1*dim2*dim3);//注意这里必须要将类型写对否则结果不正确
    // float * arr2_data = arr3.data<float>();
    // memcpy( &results[0],const_cast<float*>(arr3.data<float>()),sizeof(float)*dim1*dim2*dim3);

    // // for(int index=0;index<16*48*144;index++)
    // // {
    // //     results[index]=arr2_data[index];
    // // }


    // show_vector(results);

    // xt::xtensor<double, 4>::shape_type shape_x = {dim0,dim1,dim2,dim3};
    // xt::xtensor<double,4> tn1 = xt::adapt(results,shape_x);

    // std::cout<<tn1<<std::endl;


    // std::cout<<"#######################"<<std::endl;

    // xt::xtensor<double,3> batch_outputs =xt::view(tn1, 0,xt::all(),xt::all(),xt::all());
    // std::cout<<batch_outputs<<std::endl;


    // //python code tp_mask = batch_outputs[:, 7:, :, :]

    // xt::xtensor<double,4> tp_mask = xt::view(tn1,xt::all(),xt::range(7,16),xt::all(),xt::all());

    // std::cout<<"tp_mask"<<tp_mask<<std::endl;

    // std::cout<<"tp_mask shape "<<xt::adapt(tp_mask.shape())<<std::endl;


    // //SCORE_THRESH = 0.2
    // //LENGTH_THRESH = 5
    // //TOPK_N = 150
    // std::vector<lines_single> result_lines=deccode_lines_xtensor(tp_mask,0.2,5,150,3);
    //test xtensor

    std::vector<lines_single> result=mlsd_postprocess_xtensor(dim1,dim2,dim3,arr3.data<float>());






    
    


    
    // Fastor::Tensor<float,1,16,80,240> output_80_240;
    // Fastor::Tensor<float,1,16,160,480> output_320_960;

//     int size=16*80*240;

    

//     if(size==16*80*240)
//     {   
//         Fastor::Tensor<float,1,16,80,240> output_80_240;
        
//     }
//     else if(size==16*160*480)
//     {
//         Fastor::Tensor<float,1,16,160,480> output_320_960;
        
//     }

//     //there start
//     Fastor::Tensor<float,1,16,48,144> tn1(arr2_data);

//    // auto tn3 = Fastor::reshape<1,16,48,144>(tn1);


//     //print("test_tn1",tn1);//read ok

//     Fastor::Tensor<float,1,16,48,144> batch_outputs = tn1(0,Fastor::all,Fastor::all,Fastor::all);

//     //print("batch_outputs :",batch_outputs);

//     Fastor::Tensor<float,1,9,48,144> tp_mask = batch_outputs(Fastor::all,Fastor::seq(7,16),Fastor::all,Fastor::all);

//     //print("tp_mask :",tp_mask);

//     std::cout<<"tp_mask size: "<<tp_mask.dimension(2)<<std::endl;
//     //dimension(0) dimension(1) dimension(2) dimension(3)

//     /*decode_lines_data temp_n=deccode_lines(tp_mask, 0.2, 5.0, 150, 3.0);*/

//     decode_lines_data temp_n=mlsd_post_process_16_48_144(arr2_data);


//     std::vector<std::vector<float>> alls=temp_n.np_lines_v;

//     for(int i=0;i<alls.size();i++)
//     {
//         for(int j=0;j<alls[i].size();j++)
//         {
//             std::cout<<alls[i][j]<<" ";
//         }
//         std::cout<<"\n";
//     }


//     Fastor::Tensor<float,1,3,3> my_matrix = {{{1,2,3},
//                                   {4,5,6},
//                                   {7,8,9}}};

//     Fastor::Tensor<float,5,5> my_matrix_new;
//     my_matrix_new.zeros();

//     my_matrix_new(Fastor::seq(1,4),Fastor::seq(1,4))=my_matrix(0,Fastor::all,Fastor::all);
//     print("my matrix_new",my_matrix_new);

//     Fastor::Tensor<double,2,3> a   = {{10,20,30},{40,50,60}};
// // create mask tensor
//     Fastor::Tensor<bool,2,3> mask = {{false,false,true},{true,false,true}};
//     Fastor::Tensor<double,2,3> new_ = a(mask)*1;
//     Fastor::Tensor<double,6,1> new_flatten=Fastor::flatten(new_);
//     print("mask after",new_flatten);

//     std::vector<int> data = {5, 16, 4, 7}; 
//     std::vector<int> index_v=sort_index(data);
    
//     show_vector(index_v);



//   Fastor::Tensor<double> my_scalar = 2; // this is a scalar (double) with a value of 2
//   // output this on the screen
//   print("my scalar",my_scalar); // print is a built-in Fastor function for printing

//   Fastor::Tensor<double,3> my_vector = {1,2,3}; // this is a vector of 3 doubles with values {1,2,3}
//   print("my vector",my_vector);
//   Fastor::Tensor<float,3,3> my_matrix = {{1,2,3},
//                                  {4,5,6},
//                                  {7,8,9}}; // this a 3x3 matrix of floats with values 1...9
//   print("my matrix",my_matrix);

//   Fastor::Tensor<int,2,2,2,2,2> array_5d; // this a 5D array of ints with dimension 2x2x2x2x2
//   array_5d.iota(1); // fill the 5d_array with sequentially ascending numbers from 1 to 2x2x2x2x2=32
//   print("my 5D array", array_5d);


}