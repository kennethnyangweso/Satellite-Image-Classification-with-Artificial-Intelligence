# Satellite-Image-Classification-with-Artificial-Intelligence

# 📌 **Business Understanding: Satellite Image Classification Project**

## 🔍 **Overview**
Satellite imagery is a powerful tool for understanding and managing land resources, urban expansion, and environmental change. However, extracting meaningful insights from these images requires significant time and expertise. By leveraging deep learning, we aim to automate the classification of satellite images into meaningful land cover categories, helping stakeholders make faster, data-driven decisions.

---


## 🧠 **Problem Statement**
Manual classification of satellite imagery is a slow and error-prone process, making it difficult for governments, urban planners, and environmental agencies to monitor land use and environmental changes efficiently. With the increasing volume of satellite data, there is a growing need for automated systems that can accurately and quickly categorize different land types from satellite images.

This project aims to develop an automated image classification model that can categorize satellite images into distinct land cover classes such as forest, agricultural land, water bodies, and residential areas.

---

## 🎯 **Objectives**
To build and deploy a machine learning model capable of:
- Automatically classifying satellite images into predefined categories
- Enabling data-driven decision-making in urban development and environmental monitoring
- Laying the foundation for a scalable, deployable application

---

## 📈 **Business Goals**
- **Urban Planning**: Identify and monitor expanding urban and industrial zones.
- **Agricultural Monitoring**: Track changes in crop fields and farming activities.
- **Environmental Monitoring**: Detect deforestation, water body shrinkage, or desertification.
- **Disaster Response**: Aid in rapid damage assessment after events like floods or wildfires.

---

## ✅ **Success Criteria**
- Achieve at least **85% classification accuracy** on validation data
- Ensure the model is lightweight enough for real-time inference
- Deploy the model in a user-friendly application (e.g., Streamlit) for use by non-technical stakeholders

## 🧠 **Modeling**
- For the modeling I opted for the deep learning models i.e
    - Convolution Neural Network (CNN)
    - Transfer Learning model which the MobileNetV2
 
### **Best Performing Model**
The best performing model was the MobileNetV2 and the metrics are as follows:

36/36 ━━━━━━━━━━━━━━━━━━━━ 16s 432ms/step - accuracy: 0.9833 - loss: 0.0437
Validation Loss: 0.0490
Validation Accuracy: 0.9822

### **Visualizing the Confusion Matrix**

![image](https://github.com/user-attachments/assets/13d27cc2-e391-460a-b975-09e2ea023c0f)

**Observations:**

Overall Performance: The true positive counts (diagonal) are generally modest, and there's a significant amount of off-diagonal values, indicating that the MobileNetV2 model struggles with distinguishing between these land cover classes. The model's performance appears to be less precise compared to what one might hope for.

**True Positives (Correct Classifications) by Class:**

cloudy: 76

desert: 47

green_area: 80

water: 80

The model performs best on 'green_area' and 'water', both having 80 correct predictions.

The model performs worst on 'desert', with only 47 correct predictions.

**False Positives and False Negatives (Misclassifications) by Class:**

Actual 'cloudy' images:

54 were predicted as 'desert'.

85 were predicted as 'green_area'.

85 were predicted as 'water'.

 ✅ # **Conclusions**

1.Transfer Learning Outperformed Custom CNNs

- The pretrained MobileNetV2 model (without fine-tuning) achieved the best results with a validation accuracy of 98.4% and the highest number of true positives per class.

- Custom CNNs, even with class balancing and augmentation, struggled to generalize and consistently misclassified samples.

2. Fine-Tuning Did Not Yield Additional Gains

- While fine-tuning slightly improved validation accuracy (to 93.2%) over the base CNNs, it underperformed the original MobileNetV2 in terms of class-level prediction quality, as shown in the confusion matrix.

- Overfitting and potential instability from unfreezing layers likely contributed to reduced generalization.

3. Class Confusion Was Pronounced in Visually Similar Classes

- Classes such as cloudy, green area, and water showed overlap due to shared textures and tones.

- Despite this, the base model handled these overlaps better than the tuned versions.

# ✅ **Recommendations**

1. Use the Base MobileNetV2 Model for Deployments ince it delivered the best results, this model should be saved and deployed for real-world use cases.

2. Deploy Using Streamlit or Flask

- Create a simple web interface for users to upload satellite images and receive class predictions in real time.

- Streamlit is recommended for rapid prototyping and demo apps.

3. Explore Grad-CAM for Model Explainability

- Implement Grad-CAM heatmaps to visualize what parts of the image the model uses for its predictions.

- This improves trust and transparency for stakeholders.

4. Improve Data Quality and Class Balance

- Consider curating more training samples for underrepresented or confused classes (e.g., “desert”).

- Use higher resolution images and additional data augmentation to further enhance robustness.

5. Extend to Multilabel or Multiscale Classification

- Satellite scenes often contain multiple features. Future work could explore multilabel classification, land-use segmentation, or region-based analysis.

# Deployment

![Satellite Classifier - Google Chrome 7_8_2025 12_59_07 PM](https://github.com/user-attachments/assets/93304977-f5fb-4658-b3ab-9045f1771e46)

![Satellite Classifier - Google Chrome 7_8_2025 12_59_19 PM](https://github.com/user-attachments/assets/54293061-bfb9-4cc9-8ba6-3b57535d24c5)
