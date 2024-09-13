function tar = warp_img(ref, deform)

[height, width] = size(ref);

ref = double(ref);

y_seq = 0 : height - 1;
x_seq = 0 : width - 1;
[x_mesh, y_mesh] = meshgrid(x_seq, y_seq);
mesh(:,:,1) = x_mesh;
mesh(:,:,2) = y_mesh;

coord = mesh + deform;

mesh = reshape(permute(mesh, [2, 1, 3]), width * height, 2);
coord = reshape(permute(coord, [2, 1, 3]), width * height, 2);
deform = reshape(permute(deform, [2, 1, 3]), width * height, 2);

[idxs, ~] = knnsearch(coord, mesh, 'k', 16);

cCoord = coord(idxs, :);
cDeform = deform(idxs, :);

cCoord = reshape(cCoord, width * height, 16, 2);
cDeform = reshape(cDeform, width * height, 16, 2);

tar = zeros(height, width);
for i = 1 : width * height
    x = cCoord(i, :, 1);
    y = cCoord(i, :, 2);
    u = cDeform(i, :, 1);
    v = cDeform(i, :, 2);

    xq = mesh(i, 1);
    yq = mesh(i, 2);

    du = griddata(x, y, u, xq, yq, 'cubic');
    dv = griddata(x, y, v, xq, yq, 'cubic');

    if isnan(du) || isnan(dv)
        continue;
    end

    ox = xq - du;
    oy = yq - dv;

    fx = floor(ox) - 3;
    ex = floor(ox) + 3;
    fy = floor(oy) - 3;
    ey = floor(oy) + 3;

    if fx < 0
        fx = 0;
        ex = 6;
    end
    if fy < 0
        fy = 0;
        ey = 6;
    end

    if ex >= width
        ex = width - 1;
        fx = ex - 6;
    end
    if ey >= height
        ey = height - 1;
        fy = ey - 6;
    end

    xx_seq = fx : ex;
    yy_seq = fy : ey;

    val = fnval(spapi({6, 6}, {yy_seq, xx_seq}, ref(yy_seq + 1, xx_seq + 1)), [oy; ox]);
    if isnan(val)
        continue
    end
    val = round(val);
    if val > 255
        val = 255;
    end
    if val < 0
        val = 0;
    end
    tar(mesh(i, 2) + 1, mesh(i, 1) + 1) = val;
end

% tar(isnan(tar) == 1) = 0;
% tar(tar > 255) = 255;
% tar(tar < 0) = 0;
    