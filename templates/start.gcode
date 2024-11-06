M201 X500 Y500 Z100 E5000 ; sets maximum accelerations, mm/sec^2
M203 X500 Y500 Z10 E60 ; sets maximum feedrates, mm / sec
M204 S500 T1000 ; sets acceleration (S) and retract acceleration (R), mm/sec^2
M205 X8.00 Y8.00 Z0.40 E5.00 ; sets the jerk limits, mm/sec
M205 S0 T0 ; sets the minimum extruding and travel feed rate, mm/sec

; printing object Bed_Leveling_-_Squishing_Boxes.stl id:0 copy 0
; stop printing object Bed_Leveling_-_Squishing_Boxes.stl id:0 copy 0

;TYPE:Custom
G90 ; use absolute coordinates
M83 ; extruder relative mode
G28 ; home all axis
G1 Z50 F240
G1 X2.0 Y10 F3000
G21 ; set units to millimeters
G90 ; use absolute coordinates
M83 ; use relative distances for extrusion