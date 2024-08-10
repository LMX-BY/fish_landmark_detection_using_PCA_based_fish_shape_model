import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2


# 红：255，0，0
# 橙: 255,125,0
# 黄：255，255，0
# 绿：0，255，0
# 蓝：0，0，255
# 靛: 0,255,255
# 紫: 255,0,255
def display_img(img_data, fig, axes):
    for i, img in enumerate(img_data):
        if type(img) == torch.Tensor:
            img = img.permute(1, 2, 0).np()
        axes[i].imshow(img)

    return fig, axes


def display_grid(x_points, y_points, ax, special_point=None):
    # plot grid
    for x in x_points:
        for y in y_points:
            ax.scatter(x, y, color="w", marker='+')

    # plot a special point we want to emphasize on the grid
    if special_point:
        x, y = special_point
        ax.scatter(x, y, color="red", marker='+')


def display_normalized_landmark_in_pts_in_img(target, img, pointColorMap):
    img_width = img.shape[1]
    img_height = img.shape[0]
    img_size = torch.Tensor([img_width, img_height,
                             img_width, img_height,
                             img_width, img_height,
                             img_width, img_height])
    for target_index, a_target in enumerate(target):
        unnormalized_target = a_target.cpu() * img_size
        cv2.circle(img, (int(unnormalized_target[0]), int(unnormalized_target[1])), 1, pointColorMap[0], 8)
        cv2.circle(img, (int(unnormalized_target[2]), int(unnormalized_target[3])), 1, pointColorMap[1], 8)
        cv2.circle(img, (int(unnormalized_target[4]), int(unnormalized_target[5])), 1, pointColorMap[2], 8)
        cv2.circle(img, (int(unnormalized_target[6]), int(unnormalized_target[7])), 1, pointColorMap[3], 8)


def display_normalized_landmark_in_polygon_in_img(target, img, color):
    img_width = img.shape[1]
    img_height = img.shape[0]
    img_size = torch.Tensor([img_width, img_height,
                             img_width, img_height,
                             img_width, img_height,
                             img_width, img_height])
    for target_index, a_target in enumerate(target):
        unnormalized_target = a_target.cpu() * img_size
        polygon_points = unnormalized_target.numpy().reshape(4, 2)
        cv2.polylines(img, np.int32([polygon_points]), True, color, thickness=2)


def display_landmark_in_polygon_in_img(target, img, color):
    for a_target in target:
        polygon_points = a_target.cpu().detach().numpy().reshape(4, 2)
        cv2.polylines(img, np.int32([polygon_points]), True, color, thickness=2)
        cv2.circle(img, (int(a_target[0]), int(a_target[1])), 5, (0,255,255), -1)
        cv2.circle(img, (int(a_target[2]), int(a_target[3])), 5, (112,25,25), -1)
        cv2.circle(img, (int(a_target[4]), int(a_target[5])), 5, (255,255,255), -1)
        cv2.circle(img, (int(a_target[6]), int(a_target[7])), 5, (255,255,0), -1)


def display_landmarks_in_pts_in_img(target, img, pointColorMap):
    for target_index, a_target in enumerate(target):
        cv2.circle(img, (int(a_target[0]), int(a_target[1])), 1, pointColorMap[0], 8)
        cv2.circle(img, (int(a_target[2]), int(a_target[3])), 1, pointColorMap[1], 8)
        cv2.circle(img, (int(a_target[4]), int(a_target[5])), 1, pointColorMap[2], 8)
        cv2.circle(img, (int(a_target[6]), int(a_target[7])), 1, pointColorMap[3], 8)


def display_landmarks_in_polygon_in_img(target, img, color):
    for target_index, a_target in enumerate(target):
        polygon_points = a_target[0:8].numpy().reshape(4, 2)
        cv2.polylines(img, np.int32([polygon_points]), True, color)


def display_normalized_bbox(bboxs, img, color):
    img_width = img.shape[1]
    img_height = img.shape[0]
    img_size = torch.Tensor([img_width, img_height, img_width, img_height])
    for a_b in bboxs:
        a_unnormalized_bbox = a_b.cpu() * img_size
        try:
            minX = max(int(a_unnormalized_bbox[0]), 0)
            minY = max(int(a_unnormalized_bbox[1]), 0)
            maxX = min(int(a_unnormalized_bbox[2]), img_width)
            maxY = min(int(a_unnormalized_bbox[3]), img_height)
            cv2.rectangle(img, (minX, minY), (maxX, maxY), color, 2)
        except Exception as e:
            pass

        # try:
        #     # cv2.rectangle(img, (int(a_unnormalized_bbox[0]), int(a_unnormalized_bbox[1])),
        #     #           (int(a_unnormalized_bbox[2]), int(a_unnormalized_bbox[3])), color, 2)
        #     # cv2.rectangle(img, (int(-64255980.), int(-2.5847e+09)),
        #     #               (int(64261004.), int(2.5847e+09)), color, 2)
        #     # cv2.rectangle(img, (int(-64255980.), int(0)),
        #     #                (int(64261004.), int(2.5847e+09)), color, 2)
        #     # cv2.rectangle(img, (int(0), int(0)),
        #     #               (int(64261004.), int(2.5847e+09)), color, 2)
        #     # cv2.rectangle(img, (int(0), int(0)),
        #     #                (int(0), int(2.5847e+09)), color, 2)
        #     cv2.rectangle(img,  (int(0), -2584700000),
        #                    (int(0), int(0)), color, 2)
        # except Exception as e:
        #     print(e)
        #     print(type(int(a_unnormalized_bbox[0])))
        #     print(type(int(a_unnormalized_bbox[1])))
        #     print(type(int(a_unnormalized_bbox[2])))
        #     print(type(int(a_unnormalized_bbox[3])))
        #     print("test")


def display_bbox(bboxs, img, color):
    for a_b in bboxs:
        cv2.rectangle(img, (int(a_b[0]), int(a_b[1])),
                      (int(a_b[2]), int(a_b[3])), color, 2)


def display_bbox_landmarks_with_mark_num_16(data, img, shape_color, text_color1, text_color2):
    for index, a_data in enumerate(data):
        cv2.rectangle(img, (int(a_data[0]), int(a_data[1])),
                      (int(a_data[2]), int(a_data[3])), shape_color, 2)
        text = f'{index}'
        # cv2.putText(img, text, (int(a_data[0]), int(a_data[1])),
        #             cv2.FONT_HERSHEY_DUPLEX, 0.6, text_color1)
        if a_data[4] != -1:
            a_data_landmarks = np.zeros(8)
            a_data_landmarks[0:2] = a_data[4:6]
            a_data_landmarks[2:4] = a_data[7:9]
            a_data_landmarks[4:6] = a_data[10:12]
            a_data_landmarks[6:8] = a_data[13:15]
            landmark_center = (a_data_landmarks[0:2] + a_data_landmarks[2:4] + a_data_landmarks[4:6] + a_data_landmarks[
                                                                                                       6:8]) / 4
            # cv2.putText(img, text, (int(landmark_center[0]), int(landmark_center[1])),
            #             cv2.FONT_HERSHEY_DUPLEX, 0.6, text_color1)

            cv2.polylines(img, [np.int32(a_data_landmarks).reshape(4, 2)], True, shape_color, 2)

            cv2.putText(img, f'm1', (int(a_data_landmarks[0]), int(a_data_landmarks[1])),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, text_color2)
            cv2.putText(img, f'm2', (int(a_data_landmarks[2]), int(a_data_landmarks[3])),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, text_color2)
            cv2.putText(img, f'm3', (int(a_data_landmarks[4]), int(a_data_landmarks[5])),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, text_color2)
            cv2.putText(img, f'm4', (int(a_data_landmarks[6]), int(a_data_landmarks[7])),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, text_color2)


def display_bbox_landmarks_with_mark_num_12(data, img, shape_color, text_color1, text_color2):
    for index, a_data in enumerate(data):
        cv2.rectangle(img, (int(a_data[0]), int(a_data[1])),
                      (int(a_data[2]), int(a_data[3])), shape_color, 2)
        text = f'{index}'
        cv2.putText(img, text, (int(a_data[0]), int(a_data[1])),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, text_color1)
        if a_data[4] != -1:
            a_data_landmarks = np.zeros(8)
            a_data_landmarks[0:2] = a_data[4:6]
            a_data_landmarks[2:4] = a_data[6:8]
            a_data_landmarks[4:6] = a_data[8:10]
            a_data_landmarks[6:8] = a_data[10:12]
            landmark_center = (a_data_landmarks[0:2] + a_data_landmarks[2:4] + a_data_landmarks[4:6] + a_data_landmarks[
                                                                                                       6:8]) / 4
            cv2.putText(img, text, (int(landmark_center[0]), int(landmark_center[1])),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, text_color1)
            cv2.putText(img, f'm1', (int(a_data_landmarks[0]), int(a_data_landmarks[1])),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color2)
            cv2.putText(img, f'm2', (int(a_data_landmarks[2]), int(a_data_landmarks[3])),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color2)
            cv2.putText(img, f'm3', (int(a_data_landmarks[4]), int(a_data_landmarks[5])),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color2)
            cv2.putText(img, f'm4', (int(a_data_landmarks[6]), int(a_data_landmarks[7])),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color2)
            cv2.polylines(img, [np.int32(a_data_landmarks).reshape(4, 2)], True, shape_color)


def display_score(scores, pos, img, color):
    for index, a_score in enumerate(scores):
        text = "{:.4f}".format(a_score)
        cx = pos[index][0]
        cy = pos[index][1]
        cv2.putText(img, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, color)


def display_bbox_and_landmark_in_polygon_and_score(detected_results, img, shape_color, text_color):
    img_height = img.shape[0]
    img_width = img.shape[1]
    for b in detected_results:
        # if b[4] < args.vis_thres:
        #     continue
        text = "{:.4f}".format(b[4])
        try:
            b = list(map(int, b))
            b0 = max(0, int(b[0]))
            b1 = max(0, int(b[1]))
            b2 = min(int(b[2]), img_width)
            b3 = min(int(b[3]), img_height)
            #cv2.rectangle(img, (b0, b1), (b2, b3), shape_color, 2)
            cx = b0
            cy = b1 + 12
            # cv2.putText(img, text, (cx, cy),
            #             cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color)

            # # landms
            # # cv2.circle(img, (int(b[5]), int(b[6])), 1, (0, 0, 255), 4)
            # # cv2.circle(img, (int(b[7]), int(b[8])), 1, (0, 255, 255), 4)
            # # cv2.circle(img, (int(b[9]), int(b[10])), 1, (255, 0, 255), 4)
            # # cv2.circle(img, (int(b[11]), int(b[12])), 1, (0, 255, 0), 4)
            #
            # cv2.circle(img, (int(b[5]), int(b[6])), 1, (255, 0, 0), 4)
            # cv2.circle(img, (int(b[7]), int(b[8])), 1, (255, 0, 0), 4)
            # cv2.circle(img, (int(b[9]), int(b[10])), 1, (255, 0, 0), 4)
            # cv2.circle(img, (int(b[11]), int(b[12])), 1, (255, 0, 0), 4)
            # polygon_points = b[5:12].cpu().numpy().reshape(4, 2)
            test = np.int32(b[5:13]).reshape(4, 2)
            cv2.circle(img, (int(b[5]), int(b[6])), 5, (0, 255, 255), -1)
            cv2.circle(img, (int(b[7]), int(b[8])), 5, (112, 25, 25), -1)
            cv2.circle(img, (int(b[9]), int(b[10])), 5, (255, 255, 255), -1)
            cv2.circle(img, (int(b[11]), int(b[12])), 5, (255, 255, 0), -1)
            cv2.polylines(img, [np.int32(b[5:13]).reshape(4, 2)], True, shape_color, thickness=2)
        except:
            pass
        # cv2.circle(img, (b[13], b[14]), 1, (255, 0, 0), 4)


def show_key_points_8(axis, keypoints, colors=None, size=20):
    if colors == None:
        colors = ['r', 'g', 'b', 'y']
    axis.scatter(keypoints[0], keypoints[1], s=size, marker='.', c=colors[0])
    axis.scatter(keypoints[2], keypoints[3], s=size, marker='.', c=colors[1])
    axis.scatter(keypoints[4], keypoints[5], s=size, marker='.', c=colors[2])
    axis.scatter(keypoints[6], keypoints[7], s=size, marker='.', c=colors[3])


def display_key_points(key_points_set, axes, color, pointColorMap=None):
    # if type(key_points_set) == np.ndarray:
    #     key_points_tensor = torch.from_numpy(key_points_set)
    # if type(key_points_set) == list:
    key_points_tensor = torch.from_numpy(np.array(key_points_set))
    labels = ['head', 'fin_1', 'tail', 'fin_2']

    for key_points in key_points_tensor:
        # axes.fill(key_points[0:8:2], key_points[1:9:2],alpha=0,c=color)
        reshaped_points = key_points.reshape(-1, 2)
        # looped_key_points = key_points + key_points[0:2]
        # axes.plot(looped_key_points[0:8:2],looped_key_points[1:9:2],linewidth = 0.1, color = 'b')
        plt_poly = plt.Polygon(reshaped_points, linewidth=0.1, color=color, fill=False)
        axes.add_patch(plt_poly)

    axes.scatter(key_points_tensor[:, 0], key_points_tensor[:, 1], marker='.',
                 color=pointColorMap[0], label=labels[0])
    axes.scatter(key_points_tensor[:, 2], key_points_tensor[:, 3], marker='.',
                 color=pointColorMap[1], label=labels[1])
    axes.scatter(key_points_tensor[:, 4], key_points_tensor[:, 5], marker='.',
                 color=pointColorMap[2], label=labels[2])
    axes.scatter(key_points_tensor[:, 6], key_points_tensor[:, 7], marker='.',
                 color=pointColorMap[3], label=labels[3])

    axes.legend()


def display_key_points_counter_color_map(key_points_set, axes, counter_color_map, point_color_Map):
    # if type(key_points_set) == np.ndarray:
    #     key_points_tensor = torch.from_numpy(key_points_set)
    # if type(key_points_set) == list:
    key_points_tensor = torch.from_numpy(np.array(key_points_set))
    labels = ['head', 'fin_1', 'tail', 'fin_2']

    for key_points_index, key_points in enumerate(key_points_tensor):
        # axes.fill(key_points[0:8:2], key_points[1:9:2],alpha=0,c=color)
        reshaped_points = key_points.reshape(-1, 2)
        # looped_key_points = key_points + key_points[0:2]
        # axes.plot(looped_key_points[0:8:2],looped_key_points[1:9:2],linewidth = 0.1, color = 'b')
        cur_color = counter_color_map(key_points_index)
        plt_poly = plt.Polygon(reshaped_points, linewidth=1, color=cur_color, fill=False)
        axes.add_patch(plt_poly)

    axes.scatter(key_points_tensor[:, 0], key_points_tensor[:, 1], marker='.',
                 color=point_color_Map[0], label=labels[0], s=50)
    axes.scatter(key_points_tensor[:, 2], key_points_tensor[:, 3], marker='.',
                 color=point_color_Map[1], label=labels[1], s=50)
    axes.scatter(key_points_tensor[:, 4], key_points_tensor[:, 5], marker='.',
                 color=point_color_Map[2], label=labels[2], s=50)
    axes.scatter(key_points_tensor[:, 6], key_points_tensor[:, 7], marker='.',
                 color=point_color_Map[3], label=labels[3], s=50)

    axes.legend()
