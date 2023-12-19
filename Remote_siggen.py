import pyvisa as visa

rm_siggen = visa.ResourceManager()
SGC = rm_siggen.open_resource('TCPIP::192.168.0.1::INSTR')
print(SGC.query('*IDN?'))
print(SGC.query("SYST:ERR?"))