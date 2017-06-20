img = imread('./973_origin/973.jpg');
prewitt_img = edge(rgb2gray(img), 'prewitt');
imwrite(prewitt_img, './973_prewitt/973_prewitt.png');