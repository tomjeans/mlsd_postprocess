import numpy as np
import cv2
import pickle
import math

IMG_PATH = './mlsd_test.jpg'
SCORE_THRESH = 0.2
LENGTH_THRESH = 5
TOPK_N = 150
IMG_SIZE = [288, 96]  # [宽，高]


def np_max_pool2d(mat, kernel, stride, padding):
    mat_c, mat_h, mat_w = mat.shape
    #print("\n")
    #print("\n")
    #print("############inner np max pool2d")
    #print(mat_c,mat_h,mat_w)#1 48 144
    #print("############inner np max pool2d")

    print("############mat###########")
    print("\n")
    #print(mat)


    mat = np.pad(mat, ((0, 0), (padding, padding), (padding, padding)), 'constant')
    #第一维加0 横竖起来都加
    #第二维加padding 
    #第三维仍然加padding

    #padding 1
    #padding 1 
    #print("padding",padding)

    #print("\n")
    #print("\n")
    print("\n")
    #print(mat.shape)
    #print(mat)
    print("############mat###########")
    print("\n")

    #print("\n")
    #print("\n")
    #print("\n")

    #print(mat.tolist())
    
    #print("\n")
    #print("kernel[0]",kernel[0])
    #print("kernel[1]",kernel[1])

    new_h = (mat_h + 2 * padding - (kernel[1] - 1) - 1) // stride + 1
    new_w = (mat_w + 2 * padding - (kernel[0] - 1) - 1) // stride + 1

    print("new_h",new_h)
    print("new_w",new_w)

    result = np.zeros((mat_c, new_h, new_w))

    for c_idx in range(mat_c):
        #print("c_idx ",c_idx)
        for h_idx in range(0, new_h, stride):
            for w_idx in range(0, new_w, stride):
                temp = np.max(mat[c_idx][h_idx:h_idx + kernel[0], w_idx:w_idx + kernel[1]])#某个维度上的最大值
                #切片
                result[c_idx][h_idx][w_idx] = temp
    return result


def np_topk(mat, topk_n):
    print("\n")
    print("###########in np_topk#########")
    

    print("in np_topk")
    print(mat)
    print("in np_topk")
    topk_data_sort = -np.sort(-mat)[:topk_n]#sort on rows
    topk_index_sort = np.argsort(-mat)[:topk_n]#sort return position np.array([3, 1, 2]) #[1 2 0]
    
    print(topk_data_sort.shape)
    print(topk_index_sort.shape)

    print("###########in np_topk#########")

    
    
    return topk_data_sort, topk_index_sort


def deccode_lines(np_tpMap, score_thresh=0.1, len_thresh=2, topk_n=100, ksize=3):
    '''
    tpMap:
    center: tpMap[1, 0, :, :]
    displacement: tpMap[1, 1:5, :, :]
    '''
    np_b, np_c, np_h, np_w = np_tpMap.shape# (1, 9, 48, 144)
    #print("the np_tpMap shape:",np_tpMap.shape,"the size all :",np_b*np_c*np_h*np_w)
    assert np_b == 1, 'only support bsize==1'# when np_b = 1 can run 
    np_displacement = np_tpMap[:, 1:5, :, :]
    
    np_displacement_new = []
    # print("\n")
    # print("\n")
    # print("\n")
    # print("\n")
    print("#############np_tpMap#########")
    print(np_displacement.tolist())
    # print("\n")
    # print("\n")
    # print("\n")
    # print("\n")
    np_tpMap_list=np_tpMap.flatten()
    #print(len(np_displacement.flatten()))
    for j in range(1*48*144,5*48*144,1):
        np_displacement_new.append(np_tpMap_list[j])
    #print(len(np_displacement_new))
    #print(np_displacement_new)
    #print(np_displacement.flatten())
    print("#############np_tpMap#########")

    np_center = np_tpMap[:, 0, :, :]

    #(1, 4, 48, 144)

    # print("\n")
    # print("\n")
    # print("\n")

    print("#############np_center###########")
    print(np_center.shape)
    print(np_center)
    print("#############np_center###########")
    #(1, 48, 144)


    np_center_list=[]

    for jj in range(0,48*144,1):
        np_center_list.append(np_tpMap_list[jj])
    #print(len(np_center_list))
    #print(np_center_list)
    #print(np_center.flatten())


    #print("\n")
    #print("\n")
    #print("\n")
    #print("\n")

    print(-np_center)
    print("\n")
    print("\n")
    print(np.exp(-np_center))

    np_heat_list = []
    for js in range(0,len(np_center_list),1):
        np_heat_list.append(1.0/(1.0+math.exp(-np_center_list[js])))


    np_heat = 1 / (1 + np.exp(-np_center))

    print("\n")
    print("\n")

    print("###############np_heat#############")
    print(len(np_heat.flatten()))
    print(np_heat.tolist())
    print("###############np_heat#############")

    #print(len(np_heat_list))
    #print(np_heat_list)

    np_hmax = np_max_pool2d(np_heat, (ksize, ksize), stride=1, padding=(ksize - 1) // 2)
    print("###############np_hmax############")
    print(np_hmax)
    print("###############np_hmax############")



    print("\n")
    print("\n")
    print("np_hmax shape",np_hmax.shape)



    np_keep = (np_hmax == np_heat).astype('float32')

    print("\n")
    print("########np_keep##########")
    print(np_keep)
    print(np_keep.shape)
    print("########np_keep##########")

    np_heat = np_heat * np_keep

    print("\n")
    print("********np_heat********")
    print(np_heat.shape)
    print(np_heat)
    print("********np_heat********")

    print("\n")
    print(np_heat.flatten().shape)

    np_heat = np_heat.reshape(-1, )#the same as flatten
    print("\n")
    print("######np_heat#####")
    print(np_heat.tolist())
    print(np_heat.shape)
    print("######np_heat#####")


    np_heat = np.where(np_heat < score_thresh, np.zeros_like(np_heat), np_heat)# if np_heat<score_thresh zeros else np_heat
    print("\n")
    print("########np_heat after_np_where######")
    print("the score_thresh",score_thresh)
    print(np_heat.tolist())
    print("########np_heat after_np_where######")
    #if np_heat < score_thresh zero else origin 

    np_scores, np_indices = np_topk(np_heat, topk_n)

    print("##############np_scores++++++++++++++")
    print(np_scores)
    print("##############np_scores++++++++++++++")

    print("##############np_indices+++++++++++++")
    print(np_indices)
    print("##############np_indices+++++++++++++")



    np_valid_inx = np.where(np_scores > score_thresh)
    
    print("\n")
    print("##########np_valid_ind########")
    print(np_valid_inx)
    print("##########np_valid_ind########")

    #simple
    np_scores = np_scores[np_valid_inx]
    np_indices = np_indices[np_valid_inx]
    print("\n")
    print("")
    print(np_scores)

    print("\n")
    print("")
    print(np_indices)
    #simple


    np_yy = np.floor_divide(np_indices, np_w)[:, np.newaxis]#every element divide add a dim
    print("\n")
    print("##########np_yy#########")
    print(np_yy.shape)
    print(np_yy)
    print("##########np_yy#########")

    np_xx = np.fmod(np_indices, np_w)[:, np.newaxis]#every element % operator add a dim
    print("\n")
    print("##########np_xx#########")
    print(np_xx.shape)
    print(np_xx)
    print("##########np_xx#########")

    np_center_ptss = np.concatenate((np_xx, np_yy), axis=-1)

    print("\n")
    print("##########np_center_ptss#########")
    print(np_center_ptss.shape)
    print(np_center_ptss)
    print("##########np_center_ptss#########")

    #depends on column 


    np_start_point = np_center_ptss + np.squeeze(np_displacement[0, :2, np_yy, np_xx])
    print("\n")
    print("###########np_start_point##########")
    print(np_start_point)
    print("###########np_start_point##########")


    print("\n")
    print("########np_displacement########")
    np.set_printoptions(threshold=np.inf)
    print(np_displacement.shape)
    print(np_displacement[0,:2,:,:].shape)
    print("\n")
    #print(np_displacement[0, :2, np_yy, np_xx])
    
    #(0,0,np_yy,np_xx)
    #(0,1,np_yy,np_xx)
    
    #print(np_yy.shape)
    #print(np_xx.shape)
    print(np_displacement[0,2,np_yy,np_xx].shape)
    #print(np_displacement[0,2:,:,:].shape)
    #(1, 4, 48, 144)
    #(20, 1)
    #(20, 1)
    #(20, 1)
    print("########np_displacement########")

    
    np_end_point = np_center_ptss + np.squeeze(np_displacement[0, 2:, np_yy, np_xx])
    print("\n")
    print("############np_end_point###########")
    print(np_end_point)
    print("############np_end_point###########")
    print("\n")






    print("\n")
    print("###############np_end_point################")
    print(np_displacement[0,2:,np_yy,np_xx].shape)
    #(0,2,np_yy,np_xx)
    #(0,3,np_yy,np_xx)
    
    print("\n")
    print(np_end_point)
    print("###############np_end_point################")

    np_lines = np.concatenate((np_start_point, np_end_point), axis=-1)
    print("\n")
    print("##############np_lines#####################")
    print(np_lines)
    print(np_lines.shape)
    print("##############np_lines#####################")


    np_all_lens = (np_end_point - np_start_point) ** 2
    print("\n")
    print("#############np_all_lens###################")
    print(np_all_lens.shape)
    print("\n")
    print(np_all_lens)
    print("#############np_all_lens###################")


    np_all_lens = np_all_lens.sum(axis=-1)# rows add
    print("\n")
    print("#################np_all_lens_last###############")
    print(np_all_lens.shape)
    print("\n")
    print(np_all_lens)
    print("#################np_all_lens_last###############")

    np_all_lens = np.sqrt(np_all_lens)# every element sqrt
    print("\n")
    print("################np_all_lens_sqrt#####################")
    print(np_all_lens.shape)
    print(np_all_lens)
    print("################np_all_lens_sqrt#####################")


    np_valid_inx = np.where(np_all_lens > len_thresh)
    print("\n")
    print("\n")
    print("###############np_valid_ind#############")
    print(np_valid_inx)
    print(np_valid_inx)
    print("###############np_valid_ind#############")

    print("np_center_ptss before")
    print(np_center_ptss)
    print("np_center_ptss before")

    np_center_ptss = np_center_ptss[np_valid_inx]# what`s the meaning? the same as the origin
    print("\n")
    print("np_center_ptss")
    print(np_center_ptss.shape) #(20,2)
    print(np_center_ptss)
    print("np_center_ptss")

    np_lines = np_lines[np_valid_inx]
    print("\n")
    print("################np_lines###########")#get first 20 (20,4)
    print(np_lines.shape)
    print(np_lines)
    print("################np_lines###########")


    np_scores = np_scores[np_valid_inx]
    print("\n")
    print("###################np_scores###########")#get first 20 (20,1)
    print(np_scores.shape)
    print(np_scores)
    print("###################np_scores###########")

    return np_center_ptss, np_lines, np_scores


if __name__ == '__main__':

    img = cv2.imread(IMG_PATH)
    img_h, img_w, _ = img.shape
    img_show = img.copy()

    tmp_output = pickle.load(open('./mlsd.npy', 'rb'))
    
    # savein fortranarray
    w = [np.asfortranarray(x) for x in tmp_output]
    np.save('save_weights.npy',*w)
    np.save('save_weights_origin.npy',tmp_output)
    # savein fortranarray
    
    # the tmp_output shape is float32[1,16,48,144]

    batch_outputs = tmp_output[0]
    # print("#############the batch_outputs#############")
    # print(batch_outputs)
    # print("#############the batch_outputs#############")
    # batch_outputs shape is 16 48 144 

    # batch_outputs = outputs[0]
    #print(batch_outputs)
    list_rest=batch_outputs.flatten()
    #7-9
    tp_mask_new=[]
    for i in range(7*48*144,16*48*144,1):
        tp_mask_new.append(list_rest[i])
    #print("#########################")
    #print(len(tp_mask_new))#this is the tp_mask
    #print("#########################")


    tp_mask = batch_outputs[:, 7:, :, :]
    print("############the tp_mask###################")
    print(tp_mask.shape)
    #print(tp_mask.tolist())
    #print(tp_mask.flatten())
    #print(tp_mask)
    print("############the tp_mask###################")

    center_ptss, pred_lines, scores = deccode_lines(tp_mask, SCORE_THRESH, LENGTH_THRESH, TOPK_N, 3)

    pred_lines_list = []
    scores_list = []
    h, w, _ = img_show.shape
    for line, score in zip(pred_lines, scores):
        x0, y0, x1, y1 = line
        print(x0,y0,x1,y1)

        x0 = w * x0 / (IMG_SIZE[0] / 2)
        x1 = w * x1 / (IMG_SIZE[0] / 2)

        y0 = h * y0 / (IMG_SIZE[1] / 2)
        y1 = h * y1 / (IMG_SIZE[1] / 2)

        pred_lines_list.append([x0, y0, x1, y1])
        scores_list.append(score)
        # print(x0, y0, x1, y1, score)
        if score >= SCORE_THRESH:
            cv2.line(img_show, (round(x0), round(y0)), (round(x1), round(y1)), (255, 0, 255), 2)

    cv2.imshow('test', img_show)
    cv2.waitKey(0)
