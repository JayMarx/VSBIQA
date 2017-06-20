%对图片进行随机筛选，并将有损图与原图进行区分
%随机取有损图与原图的70%作为训练数据，30%作为测试数据

mat = load('./dmos.mat');
len = length(mat.dmos);
file_ptr = fopen('./all_img.txt', 'wt') ;
distortion_num = [];
origin_num = [];
for i = 1 : len     
    fprintf(file_ptr, '%06d.bmp %f\n', i, mat.dmos(i));
    %fprintf(file_ptr, '%06d.bmp\n', i);
    if mat.dmos(i) ~= 0.0
        distortion_num = [distortion_num;i]; 
    else
        origin_num = [origin_num; i];
    end
    %fprintf(file_ptr, '%06d.bmp\n', i);    %for test
end
fclose(file_ptr);

train_ptr = fopen('./train_label.txt', 'wt') ;
test_ptr = fopen('./test.txt', 'wt');

dis_len = length(distortion_num);
ori_len = length(origin_num);

train_dis_num = round(dis_len*0.7);     %取70%数据作为训练集
train_ori_num = round(ori_len*0.7);

dis_rand_list = randperm(dis_len);          %随机打乱顺序
ori_rand_list = randperm(ori_len);

for i=1:train_dis_num
    fprintf(train_ptr, '%06d.bmp %d\n',distortion_num(dis_rand_list(i)), round(mat.dmos(distortion_num(dis_rand_list(i)))));
end

for i=1:ori_len                 %取所有无损图
%for i=1:train_ori_num
    fprintf(train_ptr, '%06d.bmp %d\n',origin_num(ori_rand_list(i)), mat.dmos(origin_num(ori_rand_list(i))));
end
fclose(train_ptr);

for i=train_dis_num+1:dis_len
    fprintf(test_ptr, '%06d.bmp\n', distortion_num(dis_rand_list(i)));
end

for i=train_ori_num+1:ori_len
    fprintf(test_ptr, '%06d.bmp\n', origin_num(ori_rand_list(i)));
end
fclose(test_ptr);

