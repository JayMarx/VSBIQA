all_imgs = dir('./csiq/jpg2k/*.png');
len = length(all_imgs);
for i=1:len
    img = imread(strcat('./csiq/jpg2k/', all_imgs(i).name));
    prewitt_img = edge(rgb2gray(img), 'prewitt');
    imwrite(prewitt_img, strcat('./csiq/prewitt_images/jpg2k/', all_imgs(i).name));
end