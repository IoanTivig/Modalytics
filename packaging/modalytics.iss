; Modalytics installer (Inno Setup 6): the PyInstaller folder (packaging/modalytics.spec) as one setup program.
;
; * Installed for all users, in Program Files (administrator rights); the user library (%APPDATA%\Modalytics:
;   profiles, models, training sets, settings) is never touched by installing, updating or uninstalling.
; * Start-menu shortcuts (Modalytics, its help is inside), an optional desktop icon; .modalytics projects open with
;   Modalytics.
; * Training's PyTorch is not in the program: an unticked task downloads the training pack for this computer's GPU
;   at the end of setup (modalytics-cli --install-training, run as the user who started setup, so it goes into
;   their %LOCALAPPDATA%\Modalytics\training); otherwise Training › Train offers it when first needed.
; * Built by packaging/build.py, which passes the version and the folders:
;     ISCC /DAppVersion=0.1.0 /DNumericVersion=0.1.0.0 /DSource=build\dist\Modalytics /DOutput=build\installer
;   and /DSpanning=1 when the program does not fit in one setup file (then setup.exe + setup-N.bin go together).

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef NumericVersion
  #define NumericVersion "0.0.0.0"
#endif
#ifndef Source
  #define Source "..\build\dist\Modalytics"
#endif
#ifndef Output
  #define Output "..\build\installer"
#endif

[Setup]
AppId={{6C2E1B7A-3D2F-4E59-9A47-0C1F7D3B8E21}
AppName=Modalytics
AppVersion={#AppVersion}
AppVerName=Modalytics {#AppVersion}
AppPublisher=Ioan Cristian Tivig
AppCopyright=Copyright (C) 2026 Ioan Cristian Tivig — LGPL-3.0-or-later
VersionInfoVersion={#NumericVersion}
VersionInfoDescription=Modalytics setup
DefaultDirName={autopf}\Modalytics
DefaultGroupName=Modalytics
DisableProgramGroupPage=yes
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
LicenseFile=..\COPYING.LESSER
SetupIconFile=modalytics.ico
UninstallDisplayIcon={app}\Modalytics.exe
UninstallDisplayName=Modalytics {#AppVersion}
ChangesAssociations=yes
CloseApplications=yes
OutputDir={#Output}
OutputBaseFilename=Modalytics-{#AppVersion}-setup
Compression=lzma2/max
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=8
WizardStyle=modern
#ifdef Spanning
DiskSpanning=yes
DiskSliceSize=1900000000   ; under GitHub's 2 GiB per release file
#endif

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "trainingpack"; Description: "Download training support now (PyTorch for this computer's graphics card, up to 2.6 GB; Training can also install it later)"; GroupDescription: "Training segmentation models:"; Flags: unchecked

[InstallDelete]
; An update replaces the program whole: files of the previous version that this one no longer has are removed
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "{#Source}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Modalytics"; Filename: "{app}\Modalytics.exe"
Name: "{autodesktop}\Modalytics"; Filename: "{app}\Modalytics.exe"; Tasks: desktopicon

[Registry]
; .modalytics projects open with Modalytics
Root: HKA; Subkey: "Software\Classes\.modalytics"; ValueType: string; ValueName: ""; ValueData: "Modalytics.Project"; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\Modalytics.Project"; ValueType: string; ValueName: ""; ValueData: "Modalytics project"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\Modalytics.Project\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\Modalytics.exe,0"
Root: HKA; Subkey: "Software\Classes\Modalytics.Project\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\Modalytics.exe"" ""%1"""

[Run]
Filename: "{app}\modalytics-cli.exe"; Parameters: "--install-training"; StatusMsg: "Downloading training support (PyTorch) for this computer…"; Tasks: trainingpack; Flags: runasoriginaluser waituntilterminated
Filename: "{app}\Modalytics.exe"; Description: "{cm:LaunchProgram,Modalytics}"; Flags: nowait postinstall skipifsilent
