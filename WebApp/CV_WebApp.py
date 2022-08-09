#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import streamlit as st
import numpy as np
from PIL import Image


def brighten_image(image, amount):
    img_bright = cv2.convertScaleAbs(image, beta=amount)
    return img_bright


def blur_image(image, amount):
    blur_img = cv2.GaussianBlur(image, (11, 11), amount)
    return blur_img


def enhance_details(img):
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return hdr


def canny_edge(image):
    img_edg = cv2.Canny(image,100,200)
    return img_edg


def cartoonify(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    thre = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    bifil = cv2.bilateralFilter(image, 9, 250, 250)
    cartoon = cv2.bitwise_and(bifil, bifil, mask=thre)
    return cartoon


def segmentation(image):
    imrgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    vec = imrgb.reshape((-1,3))
    vec = np.float32(vec)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    ret,label,center = cv2.kmeans(vec,6,None,criteria,10,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    segmented = res.reshape((image.shape))
    return segmented


def main_loop():
    st.set_page_config(page_title="Welcome", page_icon=":wave:")
    st.title("Computer Vision Web App")
    st.subheader("Image filters, Edge Detection, Cartoonify, Segmentation")
    
    blur_rate = st.sidebar.slider("Blurring", min_value=0.5, max_value=3.5)
    brightness_amount = st.sidebar.slider("Brightness", min_value=-50, max_value=50, value=0)
    apply_enhancement_filter = st.sidebar.checkbox('Enhance Details')
    apply_edg_det = st.sidebar.checkbox('Edge Detection')
    apply_cartoon = st.sidebar.checkbox('Cartoonify')
    apply_segmentation = st.sidebar.checkbox('Segmentation')

    image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if not image_file:
        return None

    original_image = Image.open(image_file)
    original_image = np.array(original_image)

    processed_image = blur_image(original_image, blur_rate)
    processed_image = brighten_image(processed_image, brightness_amount)

    if apply_enhancement_filter:
        processed_image = enhance_details(processed_image)
        
    if apply_edg_det:
        processed_image = canny_edge(processed_image)
        
    if apply_cartoon:
        processed_image = cartoonify(processed_image)
        
    if apply_segmentation:
        processed_image = segmentation(processed_image)

    st.text("Imported Image")
    st.image(original_image)
    st.text("Processed Image")
    st.image(processed_image)
    

if __name__ == '__main__':
    main_loop()

