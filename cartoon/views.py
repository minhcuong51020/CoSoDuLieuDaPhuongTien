from django.shortcuts import render, redirect
from django.views import View
from .models import ImageCartoon
from skimage import feature
import os
import cv2
import numpy
# Create your views here.


def home_page(request):
    return render(request, 'index.html')


def home_start(request):
    return redirect('/home')


def load_data():
    f = open("data.csv", "r")
    # đọc dữ liệu trong file data.csv
    models = f.readlines()
    f.close()
    return models


def feature_extraction():
    data_folder ="media\\datass"
    if (not os.path.exists("data.csv")): open("data.csv", "x")
    f = open("data.csv", "w")
    # lấy các folder trong media/datass
    for folder in os.listdir(data_folder):
        current_path = os.path.join(data_folder, folder)
        # lấy ra các file trong folder tương ứng
        for file in os.listdir(current_path):
            # lấy đường dẫn của file ảnh trong tập ảnh đã có
            path = os.path.join(current_path, file)
            # đọc ảnh
            img = cv2.imread(path)
            # lấy ra vector đặc trưng về màu của ảnh
            feature1 = get_vector_histogram(img)
            # lấy ra vector hình dạng về màu của ảnh
            feature2 = get_vector_shape(img)
            # nối 2 vector lại với nhau thanh vector có 345 phần tử
            fx = numpy.concatenate((feature1, feature2))
            cv2.normalize(fx, fx)
            # ghi tên và vector đặc trưng vào file data.csv
            f.write("%s,%s\n" % (path, ",".join(str(v) for v in fx)))
    f.close()


def get_vector_histogram(img):
    #Chuyển ảnh thành 32x32 px
    img = cv2.resize(img, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #calcHist(ảnh đầu vào, kênh màu, mặt nạ, số bins, miền giá trị)
    #trả về mảng 8 chiều mỗi chiều 1 phần tử
    histH = cv2.calcHist([img], [0], None, [7], (0, 256), accumulate=True)
    histS = cv2.calcHist([img], [1], None, [11], (0, 256), accumulate=True)
    histV = cv2.calcHist([img], [2], None, [3], (0, 256), accumulate=True)
    # duỗi các histH,S,V thành các vector 1 chiều và nối vào với nhau
    feature_histogram = numpy.concatenate((histH.flatten(), histS.flatten(), histV.flatten()))
    cv2.normalize(feature_histogram, feature_histogram)
    # ma tran 21 phan tu
    return feature_histogram


def get_vector_shape(img):
    # Chuyển ảnh thành 32x32 px
    img = cv2.resize(img, (32, 32))
    # Chuyển ảnh từ RGB sang Ảnh xám
    # CT chuyển: Giá trị điểm ảnh = ( Điểm ảnh R + Điểm ảnh G + Điểm ảnh B ) / 3
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # orientations: số bin
    # pixels_per_cell: 1 cell
    # cells_per_block: 1 block gồm 4 cell
    feature_shape = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2")
    #ma tran 324 phan tu
    return feature_shape.flatten()


class ResultPage(View):
    def get(self, request):
        return redirect('/home')

    def post(self, request):
        # trích xuất là lấy ra các đặc trưng của ảnh trong kho
        feature_extraction()
        featuresData = load_data()
        # nhận ảnh ảnh từ người dùng
        imageInput = request.FILES['fileInput']
        imageObject = ImageCartoon(img=imageInput)
        # lưu ảnh người dùng gửi lên vào thư mục uploads của media
        imageObject.save()
        # đường dẫn ảnh của người dùng vừa gửi lên sau khi đã được lưu
        imageUrl = 'media' + '/' + imageObject.img.name
        # đọc ảnh của người dùng vừa gửi
        imageUse = cv2.imread(imageUrl)
        # trích rút đặc trưng màu sắc ảnh người dùng vừa gửi lên
        f1 = get_vector_histogram(imageUse)
        # trích rút đặc trưng hình dạng ảnh người dùng vừa gửi lên
        f2 = get_vector_shape(imageUse)
        # nối 2 vector màu sắc và hình dạng lại thanh 1 vector duy nhất
        features = numpy.concatenate((f1, f2))
        cv2.normalize(features, features)
        list_distantce = []
        for i in featuresData:
            # bỏ các khoảng trắng phía bên phải
            i = i.rstrip()
            # cắt chuỗi thành 1 mảng theo dấu ,
            i = i.split(",")
            # chuyển các giá trị sô thành 1 mảng các số thực
            j = numpy.array(i[1:]).astype(numpy.float32)
            # dùng euclid tính khoảng cách từ vector đặc trưng ảnh người dùng gửi lên
            # với vector đặc trưng của tập ảnh đã có trong hệ thống
            dist = [i[0][6:None], (numpy.linalg.norm(j - features))]
            # đưa tên và khoảng cách đã tính vào 1 mảng
            list_distantce.append(dist)
        # sắp xếp khoảng cách từ nhỏ đến lớn
        list_distantce.sort(key=lambda x: x[1], reverse=False)
        list_cartoon_image = []
        print("-----------------------------")
        for i in list_distantce[0:10]:
            # lấy ra đường dẫn của 6 ảnh có khoảng cách nhỏ nhất
            imgObj = ImageCartoon(None, i[0])
            # đưa 6 ảnh vào mảng để chuyển sang template
            list_cartoon_image.append(imgObj)
        # chuyển dữ liệu sang template
        return render(request, 'result.html', {'listImage' : list_cartoon_image})