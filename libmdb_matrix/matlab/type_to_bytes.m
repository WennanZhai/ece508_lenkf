function bits = type_to_bytes(type)

if strcmp(type, 'double') || strcmp(type, 'float64')
    bits = 8;
elseif strcmp(type, 'float') || strcmp(type, 'float32')
    bits = 4;
else
    assert(0)
end