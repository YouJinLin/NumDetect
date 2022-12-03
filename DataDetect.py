import cv2
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = 16,
                kernel_size = 5,
                stride = 1, 
                padding = 2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x

def Detect(Input):
    Input = cv2.resize(Input, (28, 28))
    testimg = torch.tensor(Input)
    testimg = torch.unsqueeze(testimg, dim=0)
    testimg = torch.unsqueeze(testimg, dim=0)
    testimg = testimg.to(torch.float32)

    cnn = torch.load(r'D:/python/Numbers/numbers.pth')
    Output, _ = cnn(testimg)
    predict = torch.max(Output, dim=1)[1].data.numpy()
    return predict



Image = cv2.imread(r'D:/python/Numbers/num.png')
img = cv2.cvtColor(Image, cv2.COLOR_RGB2GRAY)

imgCanny = cv2.Canny(img, 100, 200)
img = cv2.dilate(imgCanny, (5, 5), iterations=3)

contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for ctr in contours:
    cv2.drawContours(Image, ctr, -1, (0, 0, 255), 3)
    x, y, w, h = cv2.boundingRect(ctr)
    # 擷取圖片
    img_digit = img[y:y+h, x:x+w]

    r = max(w, h)
    y_pad = ((w - h) // 2 if w > h else 0) + r // 5
    x_pad = ((h - w) // 2 if h > w else 0) + r // 5
    img_digit = cv2.copyMakeBorder(img_digit, top=y_pad, bottom=y_pad, left=x_pad, right=x_pad, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    result = Detect(img_digit)
    print(f"{result[0]}")
    cv2.rectangle(Image, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv2.putText(Image, f"{result[0]}", (x-5, y), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
cv2.imshow("digit", img_digit)
cv2.imshow("img", img)
cv2.imshow("Image", Image)


if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()