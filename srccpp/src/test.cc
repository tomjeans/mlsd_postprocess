#include "cnpy.h"
#include <complex>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <iostream>
#include <Fastor/Fastor.h> 
#include <math.h>

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




Fastor::Tensor<float,1,48,144> np_max_pool2d(Fastor::Tensor<float,1,48,144> mat, std::vector<int> kernel, int stride, int padding)
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


n_topk_res np_topk(Fastor::Tensor<float,48*144,1> mat,int topk_n)
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


decode_lines_data deccode_lines(Fastor::Tensor<float,1,9,48,144> np_tpMap, float score_thresh=0.1, float len_thresh=2.0, float topk_n=100.0, float ksize=3.0)
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

Fastor::Tensor<float,1,48,144> np_hmax = np_max_pool2d(np_heat, ksize_, 1, 1);



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

n_topk_res np_scores_with_np_indices =np_topk(np_heat_after_choice,topk_n);

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










int main()
{
    //int Nx=16;
    //int Ny=48;
    //int Nz=144;
    //cnpy::npz_t arr = cnpy::npz_load("mlsd.npz");
    cnpy::NpyArray arr2 = cnpy::npy_load("save_weights.npy");
    std::cout<<"the arr shape size "<<arr2.shape.size()<<" shape[0] "<<arr2.shape[0]<<" shape[1] "<<arr2.shape[1]<<" shape[2] "<<arr2.shape[2]<<" shape[3] "<<arr2.shape[3]<<std::endl;
    cnpy::NpyArray arr3 =cnpy::npy_load("save_weights_origin.npy");
    std::cout<<"the arr shape size3 "<<arr3.shape.size()<<" shape[0] " <<arr3.shape[0]<<" shape[1] "<<arr3.shape[1]<<" shape[3] "<<arr3.shape[2]<<" shape[3] "<<arr3.shape[3]<<" shape[4] "<<arr3.shape[4]<<std::endl;

    std::vector<float> results(16*48*144);//注意这里必须要将类型写对否则结果不正确
    float * arr2_data = arr3.data<float>();
    //memcpy( &results[0],const_cast<double*>(arr2.data<double>()),sizeof(double)*16*48*144);
    for(int i=0;i<16*48*144;i++)
    {
        results.push_back(arr2_data[i]);
    }
    
    //show_vector(results);


    

    Fastor::Tensor<float,1,16,48,144> tn1(arr2_data);


    //print("test_tn1",tn1);//read ok

    Fastor::Tensor<float,1,16,48,144> batch_outputs = tn1(0,Fastor::all,Fastor::all,Fastor::all);

    //print("batch_outputs :",batch_outputs);

    Fastor::Tensor<float,1,9,48,144> tp_mask = batch_outputs(Fastor::all,Fastor::seq(7,16),Fastor::all,Fastor::all);

    //print("tp_mask :",tp_mask);

    std::cout<<"tp_mask size: "<<tp_mask.dimension(2)<<std::endl;
    //dimension(0) dimension(1) dimension(2) dimension(3)

    decode_lines_data temp_n=deccode_lines(tp_mask, 0.2, 5.0, 150, 3.0);


    std::vector<std::vector<float>> alls=temp_n.np_lines_v;

    for(int i=0;i<alls.size();i++)
    {
        for(int j=0;j<alls[i].size();j++)
        {
            std::cout<<alls[i][j]<<" ";
        }
        std::cout<<"\n";
    }


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