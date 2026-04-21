; Inno Setup script for the Votrax SC-01A Workbench (Windows installer).
;
; Prerequisites on the build machine:
;   1. Inno Setup 6 installed (https://jrsoftware.org/isinfo.php)
;   2. A completed PyInstaller build in dist\VotraxWorkbench\
;      (from the repo root: python -m PyInstaller packaging/votrax-gui.spec)
;
; Build the installer from the repo root with:
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\installer.iss
;
; Output: dist\VotraxWorkbenchSetup-<version>.exe

#define MyAppName       "Votrax SC-01A Workbench"
#define MyAppShortName  "VotraxWorkbench"
#define MyAppVersion    "0.2.0"
#define MyAppPublisher  "dengopaiv"
#define MyAppExeName    "VotraxWorkbench.exe"

[Setup]
; A stable AppId keeps upgrades / uninstalls correct. GUID generated once; do
; not change it for this product line.
AppId={{5C3F0D1A-7A3E-4D6B-9A7A-2B2C3D4E5F60}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppVerName={#MyAppName} {#MyAppVersion}
DefaultDirName={autopf}\{#MyAppShortName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
ArchitecturesAllowed=x64compatible
; Output the installer into the dist/ directory next to the PyInstaller build.
OutputDir=..\dist
OutputBaseFilename=VotraxWorkbenchSetup-{#MyAppVersion}
UninstallDisplayIcon={app}\{#MyAppExeName}
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=commandline dialog

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; \
    GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
; Recursively copy the entire PyInstaller one-dir output.
Source: "..\dist\VotraxWorkbench\*"; DestDir: "{app}"; \
    Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; \
    Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; \
    Flags: nowait postinstall skipifsilent
