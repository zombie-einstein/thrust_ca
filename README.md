# thrust_ca
## 1-D Cellular Automata implmented using thrust

Runs 1-D cellular automata on GPU using thrust, saves results visually to PNG file. Requires png++ package

Compile using:
```
nvcc -c ca.cu run_ca.cu `libpng-config --cflags`
```
And link using:
```
nvcc -o <executable_name> ca.o run_ca.o `libpng-config --ldflags`
```

Command line arguments are:
```
./<executable_name> <ca_array-size> <number_of_steps> <number_of_states> <rule_number>
```