import os
import sys

sys.path.append('stylegan2_ada')
import os.path as osp
import random
import threading
from array import array

import dearpygui.dearpygui as dpg
import numpy as np

from backend import UI_Backend

model = UI_Backend(device='cpu')

def generate_image(sender, app_data, user_data):
    checked = dpg.get_value('seed_random')
    if checked:
        seed = int(random.randint(0, 65536))
        dpg.set_value('seed', value=seed)
    else:
        seed = dpg.get_value('seed')
    image = model.gen_img(seed)
    if image is not None:
        update_image(image)

def update_image(new_image):
    # Convert image data (rgb) to raw_data (rgba)
    for i in range(0, image_pixels):
        rd_base, im_base = i * rgba_channel, i * rgb_channel
        raw_data[rd_base:rd_base + rgb_channel] = array(
            'f', new_image[im_base:im_base + rgb_channel]
        )

dpg.create_context()
dpg.create_viewport(title='DragGAN-UI', width=800, height=650)

def change_device(sender, app_data):
    model.change_device(app_data)
    pass

def weight_selected(sender):
    dpg.show_item("weight selector")
    
def seed_checkbox_pressed(sender):
    checked = dpg.get_value('seed_random')
    if checked:
        dpg.disable_item('seed')
    else:
        dpg.enable_item('seed')
    

# 定义模型参数窗口
width, height = 260, 200
posx, posy = 0, 0
with dpg.window(
    label='Network Setting', width=width, height=height, pos=(posx, posy),
    no_move=True, no_close=True, no_collapse=True, no_resize=True,
):
    dpg.add_text('device:', pos=(5, 20))
    dpg.add_combo(
        ('cpu', 'cuda', 'mps'), default_value='cpu', width=60, pos=(70, 20),
        callback=change_device,
    )
    dpg.add_text('weight:', pos=(5, 40))

    # 添加权重选择窗口
    def select_cb(sender, app_data):
        selections = app_data['selections']
        if selections:
            for fn in selections:
                # model.model_path = selections[fn]
                model.load_ckpt(selections[fn])
                dpg.set_value('weight_name', osp.basename(model.model_path))
                break

    def cancel_cb(sender, app_data):
        ...

    with dpg.file_dialog(
        directory_selector=False, show=False, callback=select_cb, id='weight selector',
        cancel_callback=cancel_cb, width=700 ,height=400
    ):
        dpg.add_file_extension('.*')

    dpg.add_button(
        label="browse", callback=weight_selected,
        pos=(70, 40),
    )
    dpg.add_text('', tag='weight_name', pos=(125, 40))
    
    # 随机种子配置
    dpg.add_text('seed:', pos=(5, 60))
    dpg.add_input_int(
        label='', width=100, pos=(70, 60), tag='seed', default_value=0,
    )
    dpg.add_checkbox(label='random seed', tag='seed_random',callback=seed_checkbox_pressed, pos=(70, 80))
    
    # 生成图像
    dpg.add_button(label="generate", pos=(70, 100), callback=generate_image)

# 定义显示图像的窗口
texture_format = dpg.mvFormat_Float_rgba
image_width, image_height, rgb_channel, rgba_channel = 512, 512, 3, 4
image_pixels = image_height * image_width
raw_data_size = image_width * image_height * rgba_channel
raw_data = array('f', [1] * raw_data_size)
with dpg.texture_registry(show=False):
    dpg.add_raw_texture(
        width=image_width, height=image_height, default_value=raw_data,
        format=texture_format, tag="image"
    )

image_posx, image_posy = 2 + width, 0
with dpg.window(
    label='Image', pos=(image_posx, image_posy), tag='Image Win',
    no_move=True, no_close=True, no_collapse=True, no_resize=True,
):
    dpg.add_image("image", show=True, tag='image_data', pos=(10, 30))

# 这里实现DragGAN相关的交互部件
# global parameters
add_point = 0 
point_color = [(1, 0, 0), (0, 0, 1)]
points, steps = [], 0
dragging = False

# 在图像上显示用户指定的点
def draw_point(x, y, color):
    x_start, x_end = max(0, x - 2), min(image_width, x + 2)
    y_start, y_end = max(0, y - 2), min(image_height, y + 2)
    for x in range(x_start, x_end):
        for y in range(y_start, y_end):
            offset = (y * image_width + x) * rgba_channel
            raw_data[offset:offset + rgb_channel] = array('f', color[:rgb_channel])

# 每点击一次add point按钮，可以在图片上指定一组起始点和目标点
def select_point(sender, app_data):
    global add_point, points
    if add_point <= 0: return
    ms_pos = dpg.get_mouse_pos(local=False)
    id_pos = dpg.get_item_pos('image_data')
    iw_pos = dpg.get_item_pos('Image Win')
    ix = int(ms_pos[0]-id_pos[0]-iw_pos[0])
    iy = int(ms_pos[1]-id_pos[1]-iw_pos[1])
    draw_point(ix, iy, point_color[add_point % 2])
    points.append(np.array([ix, iy]))
    print(points)
    add_point -= 1

def dragging_thread():
    global points, steps, dragging
    # 将用户指定的点分成初始点和目标点
    init_pts = []
    tar_pts = []
    for i in range(0, len(points), 2):
        init_pts.append(points[i])
        tar_pts.append(points[i+1])
    init_pts = np.vstack(init_pts)[:, ::-1].copy()
    tar_pts = np.vstack(tar_pts)[:, ::-1].copy()
    
    # 准备迭代优化
    model.prepare_to_drag(init_pts)

    while (dragging):
        # 迭代一次
        status, ret = model.drag(init_pts, tar_pts)
        if status:
            init_pts, _, image = ret
        else:
            dragging = False
            return
        # 显示最新的图像
        update_image(image)
        for i in range(init_pts.shape[0]):
            draw_point(int(init_pts[i][1]), int(init_pts[i][0]), point_color[0])
        for i in range(tar_pts.shape[0]):
            draw_point(int(tar_pts[i][1]), int(tar_pts[i][0]), point_color[1])

        steps += 1
        dpg.set_value('steps', f'steps: {steps}')

# posy += height + 2
drag_posx, drag_posy = posx, posy + height + 2
with dpg.window(
    label='Drag Setting', width=width, height=height, pos=(drag_posx, drag_posy),
    no_move=True, no_close=True, no_collapse=True, no_resize=True,
):
    def add_point_callback():
        global add_point
        add_point += 2
    def reset_point_callback():
        global points
        points = []
    def start_callback():
        global dragging
        if dragging: return # 如果正在drag, 则直接退出
        dragging = True
        threading.Thread(target=dragging_thread).start()
    def stop_callback():
        global dragging
        dragging = False
    
    dpg.add_text('drag', pos=(5, 20))
    # 增加point
    dpg.add_button(label="add point", width=80, pos=(70, 20), callback=add_point_callback)
    # 重置point
    dpg.add_button(label="reset point", width=80, pos=(155, 20), callback=reset_point_callback)
    # 开始运行drag过程
    dpg.add_button(label="start", width=80, pos=(70, 40), callback=start_callback)
    # 手动停止drag过程
    dpg.add_button(label="stop", width=80, pos=(155, 40), callback=stop_callback)
    # 动态显示当前已经迭代的步数
    dpg.add_text('steps: 0', tag='steps', pos=(70, 60))

# 绑定回调函数
with dpg.item_handler_registry(tag='double_clicked_handler'):
    dpg.add_item_double_clicked_handler(callback=select_point)
dpg.bind_item_handler_registry("image_data", "double_clicked_handler")


dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()