#!/usr/bin/env python3

import getopt, sys, re


def usage(argv0):
    print(argv0 + ": <-o output filename> [-q] [-a line] <header_1> ... <header_n>")
    print("-a: Add line to the top of the generated header file")


options = "ha:o:"

try:
    opts, args = getopt.gnu_getopt(sys.argv[1:], options)
except getopt.GetoptError:
    # print help information and exit:
    usage(sys.argv[0])
    sys.exit(2)


out_filename = ""
additional_lines = []

for o, a in opts:
    if o == "-h":
        usage(sys.argv[0])
        sys.exit(0)
    if o == "-o":
        out_filename = a
    if o == "-a":
        additional_lines = additional_lines + [a]

if out_filename == "":
    print("Error: output filename must be specified with the -o option")
    sys.exit(1)


try:
    out_fid = open(out_filename, "w")
except (IOError, (errno, strerror)):
    print("Error opening file: " + out_filename)
    print("Reason: " + strerror)
    sys.exit(3)

out_def_name = out_filename.upper()
out_def_name = out_def_name.replace(".", "_")

out_fid.write("#ifndef " + out_def_name + "\n")
out_fid.write("#define " + out_def_name + "\n")

for line in additional_lines:
    out_fid.write("\n")
    out_fid.write(line)
out_fid.write("\n")


# process each header file
for header_filename in args:
    try:
        header_fid = open(header_filename, "r")
    except (IOError, (errno, strerror)):
        print("Error opening file: " + header_filename)
        print("Reason: " + strerror)
        sys.exit(3)

    out_fid.write("\n")
    out_fid.write("/* " + header_filename + " */\n")

    def_name = header_filename.upper()
    def_name = def_name.replace(".", "_")

    ifndef_regex = re.compile("#ifndef\s+" + def_name)
    define_regex = re.compile("#define\s+" + def_name)
    if_regex     = re.compile("#if")
    endif_regex  = re.compile("#endif")
    include_local_regex = re.compile("#include\s+\"\w+\.h\"")

    if_count = 0

    for header_line in header_fid.readlines():
        if ifndef_regex.search(header_line) != None:
            if_count += 1
        elif define_regex.search(header_line) != None:
            pass
        elif include_local_regex.search(header_line) != None:
            pass
        elif if_regex.search(header_line) != None:
            if_count += 1
            out_fid.write(header_line)
        elif endif_regex.search(header_line) != None:
            if_count -= 1
            if (if_count != 0):
                out_fid.write(header_line)
        else:
            out_fid.write(header_line)

out_fid.write("\n#endif\n")

out_fid.close()
