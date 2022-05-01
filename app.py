import os
from PyQt6.QtWidgets import QApplication, QWidget, QLineEdit,QPushButton,QTextEdit,QVBoxLayout
from PyQt6.QtGui import QIcon
from PyQt6 import QtWidgets
from PyQt6.QtGui import QFont,QPixmap
import tensorflow as tf
import sys
import cv2

model = tf.keras.models.load_model('my_model.h5')

class Button(QPushButton):
    def __init__(self, title, parent):
        super().__init__(title, parent)
        self.setAcceptDrops(True)


    def dragEnterEvent(self, e):
        m = e.mimeData()
        if m.hasUrls():
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        m = e.mimeData()
        if m.hasUrls():
            try:
                try:
                    os.remove(("img.jpg"))
                    img = cv2.imread(m.urls()[0].toLocalFile())
                    cv2.imwrite("img.jpg", img)
                except:
                    img = cv2.imread(m.urls()[0].toLocalFile())
                    cv2.imwrite("img.jpg", img)
            except:
                None

class main_app(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MY APP")
        self.setWindowIcon(QIcon("icon.jpg"))
        self.setFixedWidth(800)
        self.setFixedHeight(600)

        self.img_display_text = QtWidgets.QLabel(self)
        self.img_display_text.setText("Chest x ray result is :")
        self.img_display_text.setFont(QFont("Times", 13))
        self.img_display_text.move(350, 10)

        self.res_text = QtWidgets.QLabel(self)
        self.res_text.setText("")
        self.res_text.setFont(QFont("Times", 13))
        self.res_text.setGeometry(0, 50, 120, 25)
        self.res_text.move(519, 7)

        self.main_display = QtWidgets.QLabel(self)
        self.pixmap = QPixmap('display.jpg')
        self.main_display.setPixmap(self.pixmap)
        self.main_display.setScaledContents(True)
        self.main_display.setGeometry(160, 45, 600, 450)

        self.show_gen_image = QPushButton("Show my result", self)
        self.show_gen_image.pressed.connect(self.img)
        self.show_gen_image.setGeometry(100, 200, 750, 70)
        self.show_gen_image.move(20, 510)

        self.initUI()

    def initUI(self):
        button = Button("Drag and Drop \n Here", self)
        button.resize(130, 450)
        button.move(10, 45)

    def img(self):
        img = cv2.imread("img.jpg")
        resized = cv2.resize(img, (240, 240), interpolation=cv2.INTER_AREA)
        resized1 = resized[None, :, :, :]
        res = model.predict(resized1)
        if res>0.5:
            self.res_text.setText("PNEUMONIA")
            self.res_text.setStyleSheet('background-color: red;')
        else:
            self.res_text.setText("Normal")
            self.res_text.setStyleSheet('background-color: green;')

        self.pixmap = QPixmap('img.jpg')
        self.main_display.setPixmap(self.pixmap)
        self.main_display.setScaledContents(True)




app = QApplication(sys.argv)

window = main_app()
window.show()
app.exec()
