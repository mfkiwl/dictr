function deform = generate_deform(width, height, grid_size, miu, sigma)

row = floor(height / grid_size) + 1;
col = floor(width / grid_size) + 1;

rHeight = row * grid_size;
rWidth = col * grid_size;

gDeform = randn([row + 1, col + 1, 2]) * sigma + miu;

y_seq = 0 : row;
x_seq = 0 : col;

gap = 1 / grid_size;

yy_seq = 0 : gap : row - gap;
xx_seq = 0 : gap : col - gap;
[x_mesh, y_mesh] = meshgrid(xx_seq, yy_seq);

deform(:, :, 1) = interp2(x_seq, y_seq, gDeform(:, :, 1), x_mesh, y_mesh, 'cubic');
deform(:, :, 2) = interp2(x_seq, y_seq, gDeform(:, :, 2), x_mesh, y_mesh, 'cubic');

bRow = floor((rHeight - height) / 2);
bCol = floor((rWidth - width) / 2);

deform = deform(bRow:bRow + height - 1, bCol:bCol + width - 1, :);
