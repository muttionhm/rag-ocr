import os
from PIL import Image
import cv2
import pytesseract
from pdf2image import convert_from_path
import time
import threading
import PyPDF2
from PIL import Image
Image.MAX_IMAGE_PIXELS = 3300000000

def layoo(pdf_path):
  result = []
  data_ocr = {}
  pdf_path = pdf_path
  pdf = convert_from_path(pdf_path, thread_count=12,dpi=200)
  bad_count = []
  for i in range(len(pdf)):
    # temp_text=pdf_fast.pages[i].extract_text()
    reader = PyPDF2.PdfReader(open(pdf_path,'rb'))
    temp_text = reader.pages[i].extract_text()
    if len(temp_text)<100:
      bad_count.append(i)
    else:
      data_ocr[i] = temp_text



  def ocr_thread(file, id):
    # 使用Pillow打开图像
    # 使用pytesseract进行OCR
    text = pytesseract.image_to_string(file, lang='chi_sim+eng')
    # 将结果保存到data字典中，键为文件名，值为识别的文本
    data_ocr[id] = text
    return

  start = time.time()
  threads = 2
  process_num = len(bad_count)//threads
  for i in range(process_num):
    threadings = []
    for j in range(threads):
      thread = threading.Thread(target=ocr_thread, args=(pdf[bad_count[i*threads+j]], bad_count[i*threads+j]))
      threadings.append(thread)
      thread.start()
    for thread in threadings:
        thread.join()

  if len(bad_count)-process_num*threads>0:
    threadings = []
    for j in range (len(pdf)-process_num*threads):
      thread = threading.Thread(target=ocr_thread, args=(pdf[bad_count[process_num*threads+j]], bad_count[process_num*threads+j]))
      threadings.append(thread)
      thread.start()
    for thread in threadings:
      thread.join()

  for i in range(len(pdf)):
    text = pytesseract.image_to_string(pdf[i], lang='chi_sim+eng')
    data_ocr[i] = text
  for i in range(len(pdf)):
    result.append(data_ocr[i])
  m_result = ''.join(result)
  return m_result
