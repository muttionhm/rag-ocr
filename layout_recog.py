import os
from PIL import Image
import cv2
import pytesseract
from pdf2image import convert_from_path
import time
import threading
def layoo(pdf_path):
  pdf = convert_from_path(pdf_path, thread_count=12,dpi=300)
  print('finished pdf2image')
  result = []
  data_ocr = {}

  # # 加载模型
  # model = lp.models.Detectron2LayoutModel(
  # "lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config",
  # extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.6],
  # label_map={0:"Text", 1:"Title", 2:"List", 3:"Table", 4:"Figure"})
  def ocr_thread(file, id):
    # 使用Pillow打开图像
    # 使用pytesseract进行OCR
    text = pytesseract.image_to_string(file, lang='chi_sim+eng')
    # 将结果保存到data字典中，键为文件名，值为识别的文本
    data_ocr[id] = text
    return

  start = time.time()
  threads = 2
  process_num = len(pdf)//threads
  for i in range(process_num):
    threadings = []
    for j in range(threads):
      thread = threading.Thread(target=ocr_thread, args=(pdf[i*threads+j], i*threads+j))
      threadings.append(thread)
      thread.start()
    for thread in threadings:
        thread.join()

  if len(pdf)-process_num*threads>0:
    threadings = []
    for j in range (len(pdf)-process_num*threads):
      thread = threading.Thread(target=ocr_thread, args=(pdf[process_num*threads+j], process_num*threads+j))
      threadings.append(thread)
      thread.start()
    for thread in threadings:
      thread.join()

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
  for i in range(len(pdf)):
    result.append(data_ocr[i])
  m_result = ''.join(result)
  return m_result
