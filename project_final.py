from __future__ import division
import tkinter
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
import cv2, math
import tkinter.font as tkFont

from tkinter import filedialog, Button, Label, Radiobutton


flag1 = 0
window = tkinter.Tk()
window.configure(background="#F0F9F7")
window.title('Spatial filtering')
window.geometry("750x550+102+120")
o_img, gray, image_out, cv_image_out = None, None, None, None
flag = 0
fontStyle = tkFont.Font(family="Lucida Grande", size=14)
fontStyle_head = tkFont.Font(family="Lucida Grande", size=25)


class gui:

    def pixels(self,row, col, img, distance):
        return img[max(row - distance, 0):min(row + distance + 1, img.shape[0]),
               max(col - distance, 0):min(col + distance + 1, img.shape[1])].flatten()

    def min_filter(self,img,mask_size,window3):
        out_img = img.copy()
        height = img.shape[0]
        width = img.shape[1]
        for i in range(height):
            for j in range(width):
                details = self.pixels(i, j, img, mask_size)
                out_img[i, j] = min(details)

        self.cv_image_out = out_img
        self.image_out = Image.fromarray(self.cv_image_out)
        self.image_out = ImageTk.PhotoImage(self.image_out.resize((225, 225)))
        Label(window3, image=self.image_out).place(relx=.8, rely=.4, anchor="c")

    def max_filter(self, img, mask_size, window3):
        out_img = img.copy()
        height = img.shape[0]
        width = img.shape[1]
        for i in range(height):
            for j in range(width):
                details = self.pixels(i, j, img, mask_size)
                out_img[i, j] = max(details)

        self.cv_image_out = out_img
        self.image_out = Image.fromarray(self.cv_image_out)
        self.image_out = ImageTk.PhotoImage(self.image_out.resize((225, 225)))
        Label(window3, image=self.image_out).place(relx=.8, rely=.4, anchor="c")

    def median_filter(self, img, mask_size, window3):
        out_img = img.copy()
        height = img.shape[0]
        width = img.shape[1]
        for i in range(height):
            for j in range(width):
                details = self.pixels(i, j, img, mask_size)
                details.sort()
                out_img[i, j] = details[int(details.size / 2)]

        self.cv_image_out = out_img
        self.image_out = Image.fromarray(self.cv_image_out)
        self.image_out = ImageTk.PhotoImage(self.image_out.resize((225, 225)))
        Label(window3, image=self.image_out).place(relx=.8, rely=.4, anchor="c")

    def mean_filter(self, img, mask_size, window3):
        out_img = img.copy()
        height = img.shape[0]
        width = img.shape[1]
        for i in range(height):
            for j in range(width):
                pixel = self.pixels(i, j, img, mask_size)
                total = sum(pixel)
                val = total / mask_size ** 2
                out_img[i][j] = val

        self.cv_image_out = out_img
        self.image_out = Image.fromarray(self.cv_image_out)
        self.image_out = ImageTk.PhotoImage(self.image_out.resize((225, 225)))
        Label(window3, image=self.image_out).place(relx=.8, rely=.4, anchor="c")

    def convolution(self,image, kernel, average=False):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_row, image_col = image.shape
        kernel_row, kernel_col = kernel.shape
        output = np.zeros(image.shape)
        pad_height = int((kernel_row - 1) / 2)
        pad_width = int((kernel_col - 1) / 2)
        padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
        padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image
        for row in range(image_row):
            for col in range(image_col):
                output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
                if average:
                    output[row, col] /= kernel.shape[0] * kernel.shape[1]
        return output

    def dnorm(self,x, mu, sd):
        return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)

    def kernelwindow(self,size, sigma=1):
        kernel_1D = np.linspace(-(size // 2), size // 2, size)
        for i in range(size):
            kernel_1D[i] = self.dnorm(kernel_1D[i], 0, sigma)
        kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
        kernel_2D *= 1.0 / kernel_2D.max()
        return kernel_2D

    def gaussian_filter(self,image, kernel_size,window3):
        kernel = self.kernelwindow(kernel_size, sigma=int(math.sqrt(kernel_size)))
        out_img = self.convolution(image, kernel, average=True)
        self.cv_image_out = out_img
        self.image_out = Image.fromarray(self.cv_image_out)
        self.image_out = ImageTk.PhotoImage(self.image_out.resize((225, 225)))
        Label(window3, image=self.image_out).place(relx=.8, rely=.4, anchor="c")

    def enter_filter_size(self,window3,typ):
        if typ=="Max":
            Label(window3, text="Enter the size of mask and press OK", background="#F0F9F7").place(relx=.5, rely=.28, anchor="c")
            self.entry1 = tkinter.Entry(window3, width=7)
            self.entry1.place(relx=.45, rely=.32, anchor="c")
            self.button=Button(window3, text="OK", background="#F0F9F7", foreground='blue', command=lambda: self.max_filter(self.gray, int(self.entry1.get()), window3))
            self.button.place(relx=.6, rely=.32, anchor="c")
        elif typ=="Min":
            Label(window3, text="Enter the size of mask and press OK", background="#F0F9F7").place(relx=.5, rely=.28, anchor="c")
            self.entry1 = tkinter.Entry(window3, width=7)
            self.entry1.place(relx=.45, rely=.32, anchor="c")
            self.button=Button(window3, text="OK", background="#F0F9F7", foreground='blue', command=lambda: self.min_filter(self.gray, int(self.entry1.get()), window3))
            self.button.place(relx=.6, rely=.32, anchor="c")
        elif typ=="Median":
            Label(window3, text="Enter the size of mask and press OK", background="#F0F9F7",).place(relx=.5, rely=.28, anchor="c")
            self.entry1 = tkinter.Entry(window3, width=7)
            self.entry1.place(relx=.45, rely=.32, anchor="c")
            self.button=Button(window3, text="OK", background="#F0F9F7", foreground='blue', command=lambda: self.median_filter(self.gray, int(self.entry1.get()), window3))
            self.button.place(relx=.6, rely=.32, anchor="c")
        elif typ == "Mean":
            Label(window3, text="Enter the size of mask and press OK", background="#F0F9F7").place(relx=.5, rely=.50, anchor="c")
            self.entry1 = tkinter.Entry(window3, width=7)
            self.entry1.place(relx=.45, rely=.55, anchor="c")
            self.button = Button(window3, text="OK", background="#F0F9F7", foreground='blue',
                                 command=lambda: self.mean_filter(self.gray, int(self.entry1.get()), window3))
            self.button.place(relx=.6, rely=.55, anchor="c")
        elif typ == "Gaussian":
            Label(window3, text="Enter the size of mask and press OK", background="#F0F9F7").place(relx=.5, rely=.50, anchor="c")
            self.entry1 = tkinter.Entry(window3, width=7)
            self.entry1.place(relx=.45, rely=.55, anchor="c")
            self.button = Button(window3, text="OK", background="#F0F9F7", foreground='blue',
                                 command=lambda: self.gaussian_filter(self.gray, int(self.entry1.get()), window3))
            self.button.place(relx=.6, rely=.55, anchor="c")

    def Laplacain_select_Kernel(self,window3):
        Label(window3, text="Select the mask size", background="#F0F9F7").place(relx=.12, rely=.65, anchor="c")
        Radiobutton(window3, text="[3x3]", background="#F0F9F7", value=5, command=lambda: self.laplacian(self.gray, window3, 3)).place(
            relx=.12, rely=.7, anchor="c")
        Radiobutton(window3, text="[5x5]", background="#F0F9F7", value=6, command=lambda: self.laplacian(self.gray, window3, 5)).place(
            relx=.24, rely=.7, anchor="c")

    def laplacian(self,img,window3,kernal_size):

            img_out = img.copy()

            height = img.shape[0]
            width = img.shape[1]
            k=0

            if kernal_size==3:
                laplace = (1.0/9) * np.array([
                    [0,-1,0],
                    [-1,4,-1],
                    [0,-1,0]])
                x=1
            elif kernal_size == 5:
                laplace = (1.0 / 25) * np.array(
                    [[0, 0, -1, 0, 0],
                     [0, -1, -2, -1, 0],
                     [-1, -2, 16, -2, -1],
                     [0, -1, -2, -1, 0],
                     [0, 0, -1, 0, 0]])
                x=3


            for i in np.arange(2, height - 2):
                for j in np.arange(2, width - 2):
                    sum = 0
                    for k in np.arange(-2, x):
                        for l in np.arange(-2, x):
                            a = img.item(i + k, j + l)
                            w = laplace[2 + k, 2 + l]
                            sum = sum + (w * a)
                    b = sum
                    img_out.itemset((i, j), b)

            self.cv_image_out = img_out
            self.image_out = Image.fromarray(self.cv_image_out)
            self.image_out = ImageTk.PhotoImage(self.image_out.resize((225, 225)))
            Label(window3, image=self.image_out).place(relx=.8, rely=.4, anchor="c")

    def median_filter_sharpening(self,img, filter_size):
        rows = np.shape(img)[0]
        columns = np.shape(img)[1]
        half_length = int((filter_size - 1) / 2)

        pad_A = np.concatenate([np.zeros((rows, half_length)), img, np.zeros((rows, half_length))], axis=1)
        pad_A = np.concatenate(
            [np.zeros((half_length, columns + 2 * half_length)), pad_A,
            np.zeros((half_length, columns + 2 * half_length))],
            axis=0)

        out_img = np.zeros_like(img)
        for i in range(0, rows):
            for j in range(0, columns):
                out_img[i, j] = np.median(pad_A[i:filter_size + i, j:filter_size + j])

        return out_img


    def unsharp_masking(self,img,window3,k):
        smoothed = self.median_filter_sharpening(img, 3)
        mask =  img - smoothed
        mask2 = k * mask
        sharpened = img + mask2
        self.cv_image_out = sharpened
        self.image_out = Image.fromarray(self.cv_image_out)
        self.image_out = ImageTk.PhotoImage(self.image_out.resize((225, 225)))
        Label(window3, image=self.image_out).place(relx=.8, rely=.4, anchor="c")

    def HighBoost(self,img,window3):
        Label(window3, text="Select the value of k", background="#F0F9F7").place(relx=.12, rely=.65, anchor="c")
        Radiobutton(window3, text="k=2", background="#F0F9F7", value=5, command=lambda: self.unsharp_masking(self.gray, window3,2)).place(
            relx=.12, rely=.7, anchor="c")
        Radiobutton(window3, text="k=3", background="#F0F9F7", value=6, command=lambda: self.unsharp_masking(self.gray, window3,3)).place(
            relx=.24, rely=.7, anchor="c")

    def browsefunc(self,window):
        filename = filedialog.askopenfilename()

        image = cv2.imread(filename)
        image12 = image.astype('uint8')
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        self.o_img = Image.fromarray(self.gray)
        self.dis_o_img = ImageTk.PhotoImage(self.o_img.resize((300, 300)))
        self.o_img = ImageTk.PhotoImage(self.o_img.resize((225, 225)))

        Label(window, image=self.dis_o_img).place(relx=.5, rely=.3, anchor="c")


    def page3(self, window3):
        window3.withdraw()
        window4 = tkinter.Toplevel()
        window4.configure(background="#F0F9F7")
        window4.title('Image Comparision')
        window4.geometry("800x680+102+120")

        or_img4 = Label(window4, text="Original Image", background="#F0F9F7")
        or_img4.place(relx=.12, rely=.1, anchor="c")
        self.disp_org_img=Image.fromarray(self.gray)
        self.disp_org_img = ImageTk.PhotoImage(self.disp_org_img.resize((300, 300)))
        Label(window4, image=self.disp_org_img).place(relx=.20, rely=.38, anchor="c")


        re_img = Label(window4, text="Smoothened/Sharpened Image",background="#F0F9F7")
        re_img.place(relx=.75, rely=.1, anchor="c")
        self.disp_imag_out=Image.fromarray(self.cv_image_out)
        self.disp_imag_out=ImageTk.PhotoImage(self.disp_imag_out.resize((300, 300)))
        Label(window4, image=self.disp_imag_out).place(relx=.8, rely=.38, anchor="c")

        PrevPage = Button(window4, text="Back",foreground='blue', background="#F0F9F7", command=lambda: self.page2(window4))
        PrevPage.place(relx=.5, rely=.7, anchor="c")

    def page2(self, window2):
        window2.withdraw()
        window3 = tkinter.Toplevel()
        window3.configure(background="#F0F9F7")
        window3.title('Image spatial')
        window3.geometry("800x680+102+120")

        orig_image = Label(window3, text="Original Image", background="#F0F9F7")
        orig_image.place(relx=.17, rely=.1, anchor="c")
        Label(window3, image=self.o_img).place(relx=.2, rely=.4, anchor="c")


        re_img3 = Label(window3, text="After Image", background="#F0F9F7")
        re_img3.place(relx=.8, rely=.1, anchor="c")

        Label(window3, text="Smoothing (Non-Linear Filters)", background="#F0F9F7").place(relx=.5, rely=.1, anchor="c")
        Min_filter = Button(window3, text="Min Filter",foreground='blue', background="#F0F9F7",
                           command=lambda: self.enter_filter_size(window3,"Min"))
        Min_filter.place(relx=.5, rely=.15, anchor="c")

        Max_filter = Button(window3, text="Max Filter",foreground='blue', background="#F0F9F7",
                            command=lambda: self.enter_filter_size(window3,"Max"))
        Max_filter.place(relx=.5, rely=.19, anchor="c")

        Median_filter = Button(window3, text="Median Filter",foreground='blue', background="#F0F9F7",
                            command=lambda: self.enter_filter_size(window3,"Median"))
        Median_filter.place(relx=.5, rely=.23, anchor="c")

        Label(window3, text="Smoothing (Linear Filters)", background="#F0F9F7").place(relx=.5, rely=.37, anchor="c")
        Mean_filter = Button(window3, text="Mean Filter", background="#F0F9F7", foreground='blue',
                               command=lambda: self.enter_filter_size(window3,"Mean"))
        Mean_filter.place(relx=.5, rely=.42, anchor="c")
        Gaussian_filter = Button(window3, text="Gaussian Filter",foreground='blue', background="#F0F9F7",
                             command=lambda: self.enter_filter_size(window3,"Gaussian"))
        Gaussian_filter.place(relx=.5, rely=.46, anchor="c")

        mfilter = Label(window3, text="Sharpening Filters", background="#F0F9F7")
        mfilter.place(relx=.5, rely=.65, anchor="c")

        Laplacian = Button(window3, text="Laplacian",foreground='blue', background="#F0F9F7", command=lambda: self.Laplacain_select_Kernel( window3))
        Laplacian.place(relx=.5,rely=.70,anchor="c")

        unsharpening = Button(window3, text="Unsharpen",foreground='blue', background="#F0F9F7", command=lambda: self.unsharp_masking(self.gray, window3,1))
        unsharpening.place(relx=.5, rely=.74, anchor="c")

        highboostfiltering = Button(window3, text="High Boost",foreground='blue', background="#F0F9F7", command=lambda: self.HighBoost(self.gray, window3))
        highboostfiltering.place(relx=.5, rely=.78, anchor="c")

        compare = Button(window3, text="Compare Image",foreground='blue', background="#F0F9F7", command=lambda: self.page3(window3))
        compare.place(relx=.5, rely=.95, anchor="c")

        PrevPage = Button(window3, text="Back",foreground='blue', background="#F0F9F7", command=lambda: self.page1(window3))
        PrevPage.place(relx=.25, rely=.95, anchor="c")

    def page1(self, window):
        window.withdraw()
        window1 = tkinter.Toplevel()
        window1.configure(background="#F0F9F7")
        window1.title('Image spatial')
        window1.geometry("800x680+102+120")

        Label(window1, text="Select an input image to be Smoothened/Sharpened and click on Next",background="#F0F9F7",font=fontStyle).place(relx=.45, rely=.75, anchor="c")
        browsebutton = Button(window1, text="Browse",foreground='blue', background="#F0F9F7", command=lambda: self.browsefunc(window1))
        browsebutton.place(relx=.2, rely=.8, anchor="c")

        NextPage = Button(window1, text="Next",foreground='blue', background="#F0F9F7", command=lambda: self.page2(window1))
        NextPage.place(relx=.7, rely=.8, anchor="c")

    def Intro_page(self,window):
        global flag
        if flag == 0:
            self.strt_Lbl=Label(window, text="Spatial Filtering Application",background="#F0F9F7", font=fontStyle_head).place(relx=.5, rely=.4, anchor="c")
            StartApp = Button(window, text="Start Application",foreground='blue',background="#F0F9F7",command=lambda: self.page1(window))
            StartApp.place(relx=.5, rely=.8, anchor="c")
            Label(window, text="Team Processors ", background="#F0F9F7").place(relx=.9, rely=.95, anchor="c")

g = gui()
g.Intro_page(window)
window.mainloop()