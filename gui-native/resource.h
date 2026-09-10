/* resource.h - control and resource identifiers for the Votrax SC-01 GUI. */

#ifndef VOTRAX_GUI_RESOURCE_H
#define VOTRAX_GUI_RESOURCE_H

/* Controls */
#define IDC_TEXTLABEL     1000
#define IDC_TEXT          1001
#define IDC_PHONEMEMODE   1002
#define IDC_MASKLABEL     1003
#define IDC_MASK          1004
#define IDC_PRESETLABEL   1005
#define IDC_PRESET        1006
#define IDC_CLOCKLABEL    1007
#define IDC_CLOCK         1008
#define IDC_CLOCKSPIN     1009
#define IDC_SPEEDLABEL    1010
#define IDC_SPEED         1011
#define IDC_SPEEDSPIN     1012
#define IDC_INFLLABEL     1013
#define IDC_INFL          1014
#define IDC_INFLSPIN      1015
#define IDC_HELPCLOCK     1016
#define IDC_HELPPHONES    1017
#define IDC_PREVIEW       1018
#define IDC_CONVERT       1019
#define IDC_RENDER        1020

/* Private messages, posted from the playback thread. */
#define WM_APP_PLAY_DONE     (WM_APP + 1)
#define WM_APP_SYNTH_FAILED  (WM_APP + 2)

#endif /* VOTRAX_GUI_RESOURCE_H */
