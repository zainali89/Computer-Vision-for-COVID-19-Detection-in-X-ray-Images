# Computer-Vision-for-COVID-19-Detection-in-X-ray-Images

The X-ray classification project leverages advanced AI and computer vision techniques, specifically a Convolutional Neural Network (CNN), to distinguish between COVID-19 and normal lung conditions from X-ray images. Trained on a comprehensive dataset of X-ray images, the AI system detects subtle patterns and anomalies indicative of COVID-19, such as specific opacities and lung abnormalities. By automating the preliminary screening process with computer vision, the project accelerates diagnostic procedures and reduces the burden on healthcare professionals, significantly enhancing the overall response to the pandemic.


### Normal Image

![norm](https://github.com/zainali89/Computer-Vision-for-COVID-19-Detection-in-X-ray-Images/assets/75775907/132e6de4-dac0-478c-bc90-91ef8d9693f1)

### Covid-19 Image

![covid](https://github.com/zainali89/Computer-Vision-for-COVID-19-Detection-in-X-ray-Images/assets/75775907/05a7e007-5a8c-46f7-8d61-ac540db8664a)


## Training Process

The training process snapshot depicts the model's performance over ten epochs during its training on a dataset of X-ray images. Initially, in the first epoch, the model starts with a higher loss of 0.6518 and an accuracy of 61.10%, which indicates that the model is in the early stages of learning from the training data. As training progresses, there is a notable improvement in both loss and accuracy, demonstrating the model's ability to learn and adapt to the features of the dataset. By the tenth epoch, the loss significantly decreases to 0.0938, and the accuracy reaches an impressive 96.96%. This trend is an excellent indicator of the model's capability to generalize well from the training data, showcasing effective learning and optimization techniques.

![training process](https://github.com/zainali89/Computer-Vision-for-COVID-19-Detection-in-X-ray-Images/assets/75775907/7b6ae921-ddd0-457f-b2da-632d4aa2c9cf)

## Classification Report

The classification report provides detailed metrics on the model's performance in distinguishing between 'Normal' and 'COVID-19' conditions from X-ray images. It offers precision, recall, and f1-score for each class, along with the support (the number of true instances for each label). For 'Normal' cases, the model achieves a precision of 0.99 and a recall of 1.00, resulting in an f1-score of 0.99. For 'COVID-19' cases, the precision is perfect at 1.00, with a recall of 0.98 and an f1-score of 0.99. The overall accuracy of the model is remarkably high at 0.99, which is consistent across the macro and weighted averages. These metrics indicate a high level of performance by the model, suggesting that it can reliably differentiate between normal and COVID-19 affected lung conditions with high accuracy.

![repor](https://github.com/zainali89/Computer-Vision-for-COVID-19-Detection-in-X-ray-Images/assets/75775907/3ae65d77-093c-4a66-8a02-385351d32f78)


