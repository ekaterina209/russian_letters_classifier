import cv2
import numpy as np
from keras.models import load_model
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLineEdit, QLabel
from PyQt5.QtGui import QImage, QPainter, QPen, QIcon, QPaintEvent
from PyQt5.QtCore import Qt, QPoint
import sys

class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        title = "Распознавание рукописных символов"
        top = 100
        left = 100
        width = 900
        height = 500

        self.drawing = False
        self.brushSize = 8
        self.brushColor = Qt.black
        self.lastPoint = QPoint()

        self.image = QImage(578, 478, QImage.Format_RGB32)
        self.image.fill(Qt.white)

        self.nameLabel = QLabel(self)
        self.nameLabel.setText('RES:')
        self.line = QLineEdit(self)

        self.line.move(660, 198)
        self.line.resize(99, 42)
        self.nameLabel.move(590, 200)

        prediction_button = QPushButton('RECOGNITION', self)
        prediction_button.move(590, 30)
        prediction_button.resize(230, 33)
        prediction_button.clicked.connect(self.save)
        prediction_button.clicked.connect(self.predicting)

        clean_button = QPushButton('CLEAN', self)
        clean_button.move(590, 100)
        clean_button.resize(230, 33)
        clean_button.clicked.connect(self.clear)

        undo_button = QPushButton('UNDO', self)
        undo_button.move(590, 150)
        undo_button.resize(230, 33)
        undo_button.clicked.connect(self.undo)

        self.setWindowTitle(title)
        self.setGeometry(top, left, width, height)

        # Загружаем модель нейронной сети
        self.model = load_model('russian_letters_classifier.h5')

        self.history = []  # Список для хранения истории холста

    def print_letters(self, results):
        letters = "АИЙКЛМНОПРСБТУФХЦЧШЩЪЫВЬЭЮЯГДЕЁЖЗ"
        predicted_letters = ''.join([letters[result] for result in results])
        self.line.setText(predicted_letters)
        return predicted_letters

    def segment_image(self, image_path):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        letter_regions = []
        padding = 8  # Задаем размер отступа для увеличения прямоугольника вокруг буквы
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            # Обрезаем координаты, чтобы они не выходили за пределы изображения
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            # Увеличиваем прямоугольник с учетом отступа
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Добавляем увеличенные области в список
            letter_regions.append((x, y, w, h))
        letter_regions = sorted(letter_regions, key=lambda x: x[0])
        return letter_regions, image

    def predicting(self):
        letter_regions, image = self.segment_image('res.jpeg')
        predictions = []
        for region in letter_regions:
            x, y, w, h = region
            # Уменьшаем размер области с буквой, чтобы удалить зеленую рамку
            x += 2  # Уменьшаем x на 2 пикселя
            y += 2  # Уменьшаем y на 2 пикселя
            w -= 3  # Уменьшаем ширину на 4 пикселя
            h -= 3  # Уменьшаем высоту на 4 пикселя

            # Создаем изображение с белой рамкой
            border_size = 40  # Размер рамки
            bordered_image = np.ones((h + 2 * border_size, w + 2 * border_size, 3),
                                     dtype=np.uint8) * 255  # Создаем пустое белое изображение
            bordered_image[border_size:border_size + h, border_size:border_size + w] = image[y:y + h,
                                                                                       x:x + w]  # Вставляем область с буквой в центр изображения
            bordered_image = cv2.resize(bordered_image,
                                        (278, 278))  # Размер изображений должен соответствовать вашей модели
            bordered_image = np.expand_dims(bordered_image, axis=0)
            bordered_image = bordered_image / 255.0
            prediction = self.model.predict(bordered_image)
            predicted_class = np.argmax(prediction)
            predictions.append(predicted_class)
        print("Predictions:", predictions)
        self.print_letters(predictions)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) & self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(self.brushColor, self.brushSize, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            # Сохраняем текущее состояние холста в историю
            self.history.append(self.image.copy())
            # Обновляем изображение, на котором происходит рисование
            self.paintEvent(QPaintEvent(self.rect()))

    def paintEvent(self, event):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(0, 0, self.image)

    def save(self):
        self.image.save('res.jpeg')

    def clear(self):
        self.image.fill(Qt.white)
        self.update()
        # Очищаем историю холста
        self.history.clear()

    def undo(self):
        if self.history:
            # Удаляем последнее состояние из истории
            self.history.pop()
            if self.history:
                # Восстанавливаем предыдущее состояние холста
                self.image = self.history[-1].copy()
            else:
                # Если история пуста, очищаем холст
                self.image.fill(Qt.white)
            self.update()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    app.exec()
