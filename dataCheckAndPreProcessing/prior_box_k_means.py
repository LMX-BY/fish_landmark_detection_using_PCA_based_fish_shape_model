import pathlib
from utiles import utils_fish_landmark_detection
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
#from dataCheckAndPreProcessing.kmeans-anchor-boxes-master import
from utiles import iou_kmeans
user_defined_json_files_str_hu_s_and_c = ('H:/code/python/IRFishDetection2.0.0/dataset2.2/'
                                          'poly_and_used_defined_label/'
                                          'all_checked_label/user_defined_json_1.0_2.0_hu_s_and_c')
if __name__ == '__main__':
    user_defined_json_files_path = pathlib.Path(user_defined_json_files_str_hu_s_and_c)
    results_display_save_path = pathlib.Path('H:/code/python/IRFishDetection2.0.0/results/bbox_k_means')
    json_files = utils_fish_landmark_detection.get_filenames_of_path(user_defined_json_files_path)
    boxes_width_and_height_list = []
    boxes_aspect_ratio_list = []
    for a_file in json_files:
        json_content = utils_fish_landmark_detection.read_json(a_file)
        bbox_and_landmarks = json_content['points']
        for bbox_and_landmarks in bbox_and_landmarks:
            a_bbox_parameter = bbox_and_landmarks[:4]
            b1x = a_bbox_parameter[0]
            b1y = a_bbox_parameter[1]
            b2x = a_bbox_parameter[2]
            b2y = a_bbox_parameter[3]
            lt_x = min(b1x, b2x)
            lt_y = min(b1y, b2y)
            rb_x = max(b1x, b2x)
            rb_y = max(b1y, b2y)
            height = rb_y - lt_y
            width = rb_x - lt_x
            assert height > 0
            assert width > 0
            width_height_rate = width / height
            boxes_aspect_ratio_list.append(width_height_rate)
            boxes_width_and_height_list.append([width,height])
    boxes_width_and_height_np = np.array(boxes_width_and_height_list)
    boxes_aspect_ratio_np = np.expand_dims(np.array(boxes_aspect_ratio_list),1)
    boxes_width = np.expand_dims(boxes_width_and_height_np[:,0],1)

    max_width_and_height = np.max(boxes_width_and_height_np,0)
    min_width_and_height = np.min(boxes_width_and_height_np,0)
    print(f'max_width_and_height: {max_width_and_height}')
    print(f'min_width_and_height: {min_width_and_height}')

    # kmeans_width_and_height = KMeans(n_clusters=9, random_state=0, n_init="auto")
    # kmeans_width_and_height.fit(boxes_width_and_height_np)
    # cluster_centers_width_and_height = kmeans_width_and_height.cluster_centers_
    # print(f'cluster_centers_width_and_height: {cluster_centers_width_and_height}')
    #
    # kmeans_aspect_ratio = KMeans(n_clusters=3, random_state=0, n_init="auto")
    # kmeans_aspect_ratio.fit(boxes_aspect_ratio_np)
    # cluster_centers_aspect_ratio = kmeans_aspect_ratio.cluster_centers_
    # print(f'cluster_centers_aspect_ratio: {cluster_centers_aspect_ratio}')
    #
    # kmeans_width = KMeans(n_clusters=3, random_state=0, n_init="auto")
    # kmeans_width.fit(boxes_width)
    # cluster_centers_width = kmeans_width.cluster_centers_
    # print(f'cluster_centers_width: {cluster_centers_width}')
    #
    # cluster_centers_separately = []
    # for a_cluster_centers_aspect_ratio in cluster_centers_aspect_ratio:
    #     for a_cluster_centers_width in cluster_centers_width:
    #         a_height = a_cluster_centers_width / a_cluster_centers_aspect_ratio
    #         cluster_centers_separately.append([float(a_cluster_centers_width), float(a_height)])
    #
    # cluster_centers_separately_np = np.array(cluster_centers_separately)
    # print(f'cluster_centers_separately: {cluster_centers_separately_np}')
    #
    #
    #
    #
    # #display
    # fig = plt.figure(f'bbox_k_means')
    # axes = fig.add_subplot(111)
    # color_all_width_and_height = 'green'
    # color_cluster_center = 'red'
    # color_separately_center = 'blue'
    # # color_map = ['red','orange','yellow','green','blue','cyan','purple','gray','sage']
    # axes.scatter(boxes_width_and_height_np[:,0],boxes_width_and_height_np[:,1], c=color_all_width_and_height, marker='.', s=1)
    # axes.scatter(cluster_centers_width_and_height[:, 0], cluster_centers_width_and_height[:, 1],
    #              c=color_cluster_center, marker='.', s=20)
    # axes.scatter(cluster_centers_separately_np[:, 0], cluster_centers_separately_np[:, 1],
    #              c=color_separately_center, marker='.', s=20)
    #
    # fig.savefig(f'{str(results_display_save_path)}/bbox_k_means.png')
    # plt.close(fig)

    #iou_kmeans based yolov2
    CLUSTERS = 8
    out = iou_kmeans.kmeans(boxes_width_and_height_np, k=CLUSTERS)
    # extra_out = np.array([[1000,1000],[1,1]])
    # add_out = np.vstack((out,extra_out))
    print("Accuracy: {:.2f}%".format(iou_kmeans.avg_iou(boxes_width_and_height_np, out) * 100))
    # print("add Accuracy: {:.2f}%".format(iou_kmeans.avg_iou(boxes_width_and_height_np, add_out) * 100))
    print("Boxes:\n {}".format(out))
    aspect_ratio_kmeans_bbox = out[:,0]/out[:,1]
    area_kmeans_bbox = out[:,0] * out[:,1]
    print("")
    # zhidan_anchor_size = [[45.2548,22.6274],[90.5097,45.2548],[181.0193,90.5097],[362.0387,181.0193],[724.0773,362.0387],[32.0000,32.0000],[64.0000,64.0000]
    #     ,[128.0000,128.0000],[256.0000,256.0000],[512.0000,512.0000],[22.6274,45.2548],[45.2548,90.5097],[90.5097,181.0193],[181.0193,362.0387],[362.0387,724.0773]]

    # tensor([49.0044, 70.5185, 95.6183, 112.3515, 34.6513, 49.8641, 67.6123,
    #         79.4445, 24.5022, 35.2592, 47.8091, 56.1757], device='cuda:0')
    #
    # tensor([34.3031, 49.3629, 66.9328, 78.6460, 48.5119, 69.8097, 94.6573,
    #         111.2223, 68.6061, 98.7259, 133.8656, 157.2921], device='cuda:0')

    # tensor([49.0044, 69.3233, 99.2040, 39.0920, 55.3008, 79.1374, 27.6422, 39.1036,
    #         55.9586, 25.4271, 35.9701, 51.4744], device='cuda:0')
    #
    # tensor([34.3031, 48.5263, 69.4428, 43.0012, 60.8309, 87.0511, 60.8128,
    #         86.0279, 123.1089, 66.1105, 93.5222, 133.8335], device='cuda:0')

    # [52.9308, 74.8777, 107.1525, 121.3535, 39.0920, 55.3008, 79.1374,
    #         89.6255, 28.2927, 40.0238, 57.2754, 64.8662, 25.4271, 35.9701,
    #         51.4744, 58.2963], device='cuda:0')
    #
    # tensor([31.7585, 44.9266, 64.2915, 72.8121, 43.0012, 60.8309, 87.0511,
    #         98.5880, 59.4146, 84.0500, 120.2784, 136.2189, 66.1105, 93.5222,
    #         133.8335, 151.5704], device='cuda:0')

    # zhidan_anchor_size1 = [[49.0044, 34.3031], [70.5185, 49.3629], [95.6183, 66.9328], [112.3515, 78.6460],
    #                       [34.6513, 48.5119], [49.8641, 69.8097], [67.6123, 94.6573]
    #     , [79.4445, 111.2223], [24.5022, 68.6061], [35.2592, 98.7259], [47.8091, 133.8656], [56.1757, 157.2921]]

    # zhidan_anchor_size2 = [[49.0044, 34.3031], [69.3233, 48.5263], [99.2040, 69.4428], [39.0920, 43.0012],
    #                        [55.3008, 60.8309], [79.1374, 87.0511], [27.6422, 60.8128]
    #     , [39.1036, 86.0279], [55.9586, 123.1089], [25.4271, 66.1105], [35.9701, 93.5222], [51.4744, 133.8335]]
    # zhidan_anchor_size3 = [[49.0044, 34.3031], [69.3233, 48.5263], [99.2040, 69.4428], [39.0920, 43.0012],
    #                        [55.3008, 60.8309], [79.1374, 87.0511], [27.6422, 60.8128]
    #     , [39.1036, 86.0279], [55.9586, 123.1089], [25.4271, 66.1105], [35.9701, 93.5222], [51.4744, 133.8335]]
    #zhidan_anchor_size_np = np.array(zhidan_anchor_size3)
    zhidan_anchor_size_np = np.vstack(([52.9308, 74.8777, 107.1525, 121.3535, 39.0920, 55.3008, 79.1374,
            89.6255, 28.2927, 40.0238, 57.2754, 64.8662, 25.4271, 35.9701,
            51.4744, 58.2963],[31.7585, 44.9266, 64.2915, 72.8121, 43.0012, 60.8309, 87.0511,
            98.5880, 59.4146, 84.0500, 120.2784, 136.2189, 66.1105, 93.5222,
            133.8335, 151.5704])).transpose(1,0)
    print("zhidan_anchor_size Accuracy: {:.2f}%".format(iou_kmeans.avg_iou(boxes_width_and_height_np,zhidan_anchor_size_np) * 100))
    pass
