function color_list = generate_colors(n)
    color_list = cell(1,n)
    for ii=1:n
        color_list{ii} = unifrnd(0, 255, 1,3)/255
    end
end