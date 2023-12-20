# default ip: 169.254.2.20
import pyvisa as visa
import time

rm_siggen = visa.ResourceManager()
SGC = rm_siggen.open_resource("TCPIP::192.168.0.100::INSTR")
print(SGC.query("*IDN?"))
print(SGC.query("SYST:ERR?"))
print(SGC.query("SOUR:FREQ?"))

SGC.write("SOUR:POW 20 dBm")
freq_list = [65, 200, 352]

for freq in freq_list:
    SGC.write(f"SOUR:FREQ {freq} MHz")
    print(f"Current freq. {SGC.query('SOUR:FREQ?')}")
    time.sleep(3)
if freq == 352 and float(SGC.query("SOUR:POW?")) != 14.0:
    SGC.write("SOUR:POW 14 dBm")
    print(f"Current power. {SGC.query('SOUR:POW:POW?')}")
    time.sleep(3)
else:
    SGC.write("SOUR:POW 20 dBm")
    time.sleep(3)

print(float(SGC.query("SOUR:POW:POW?")))
print(SGC.query("SOUR:FREQ?"))
SGC.write("SOUR:POW 1 dBm")
# SGC.write("OUTP ON")
