function type = bytes_to_type(bytes)

if (bytes == 8)
    type = 'float64';
elseif (bytes == 4)
    type = 'float32';
else
    assert(0)
end