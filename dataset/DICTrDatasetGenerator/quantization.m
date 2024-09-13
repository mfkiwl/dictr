function Ima_q = quantization(Ima,nbit)
% Image quantization
% function Ima_q = quantization(Ima,nbit)
% Ima: image to quantize
% nbit: bit-depth
% Ima_q: quantized image
% if bit-depth <= 8, quantized image is represented as uint8
% if 8< bit-depth <= 16, quantized image is represented as uint16
% in both case, intensity range is [0,2^nbit-1]

Ima=round(Ima);

if (nbit<=8)
    Ima(Ima<0)=0;
    Ima(Ima>2^(nbit)-1)=2^nbit-1;
    Ima_q = uint8(Ima);
else
    Ima(Ima<0)=0;
    Ima(Ima>2^(nbit)-1)=2^nbit-1;
    Ima_q = uint16(Ima); 
end

