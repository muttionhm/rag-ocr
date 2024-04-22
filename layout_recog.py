import os
from PIL import Image
import cv2
import pytesseract
from pdf2image import convert_from_path
import time
def layoo(pdf_path):

  pdf = convert_from_path(pdf_path, dpi=300)
  start = time.time()
  # print(len(pdf))
  result = []

  # # 加载模型
  # model = lp.models.Detectron2LayoutModel(
  # "lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config",
  # extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.6],
  # label_map={0:"Text", 1:"Title", 2:"List", 3:"Table", 4:"Figure"})
  for page_num in range(len(pdf)):
  #   page = pdf[page_num]
  #   page.save(f'page_{page_num}.jpg')
  #   image_1 = cv2.imread(f'page_{page_num}.jpg')
  #   # 检测
  #   layout = model.detect(image_1)
  #   color_map = {
  #     'text':   'red',
  #     'title':  'blue',
  #     'list':   'green',
  #     'table':  'purple',
  #     'figure': 'pink',
  # }
  # lp.draw_box(image_1, 
  #               [b.set(id=f'{b.type}/{b.score:.2f}') for b in layout],
  #               color_map=color_map,
  #               show_element_id=True, id_font_size=10, 
  #               id_text_background_color='grey',
  #               id_text_color='white')
    # text_blocks = lp.Layout([b for b in layout if b.type == 'Text']) # 循环浏览页面上的每个文本框。
    # for block in text_blocks:
    #   segment_image = (block
    #                       .pad(left=5, right=5, top=5, bottom=5)
    #                       .crop_image(image_1))
    text = pytesseract.image_to_string(pdf[page_num], lang='chi_sim+eng+spa')
    result.append(text)
  end = time.time()
  Num = len(pdf)
  dura = end-start
  print('the time is',dura)
  m_result = ''.join(result)
  return m_result
