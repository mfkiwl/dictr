function NoisyImage = addnoise(myparamnoise, b, Image)
% add noise to an image
% function NoisyImage = addnoise(myparamnoise, b, Image)
% myparamnoise: noise parameters
% b: bit-depth
% Image: input image
% NoisyImage: output image (quantized over b bits)

dims=size(Image);

if strcmp(myparamnoise.type,'G')
    Noise = myparamnoise.nu*randn(dims);
end

if strcmp(myparamnoise.type,'S')
    VarianceNoise = myparamnoise.g*double(Image) + myparamnoise.intercept;
    VarianceNoise(VarianceNoise<0)=0;
    Noise = sqrt(VarianceNoise) .* randn(dims);
end

NoisyImage=quantization(double(Image)+Noise,b);

 