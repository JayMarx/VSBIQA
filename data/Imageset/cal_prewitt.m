all_imgs = dir('./live2/*.png');
len = length(all_imgs);
for i=1:len
    img = imread(strcat('./live2/', all_imgs(i).name));
    prewitt_img = edge(rgb2gray(img), 'prewitt');
    imwrite(prewitt_img, strcat('./prewitt_images/', all_imgs(i).name));
end
