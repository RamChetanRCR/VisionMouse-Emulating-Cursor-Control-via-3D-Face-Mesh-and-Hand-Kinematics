[Setup]
AppName=Eye & Hand Tracker
AppVersion=4.0
Publisher=Ram Chetan RCR
DefaultDirName={autopf}\EyeHandTracker
DefaultGroupName=Eye & Hand Tracker
OutputDir=Output
OutputBaseFilename=EyeHandTracker_Setup
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=lowest

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "dist\MonitorTracking\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Eye & Hand Tracker"; Filename: "{app}\MonitorTracking.exe"
Name: "{group}\{cm:UninstallProgram,Eye & Hand Tracker}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Eye & Hand Tracker"; Filename: "{app}\MonitorTracking.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\MonitorTracking.exe"; Description: "{cm:LaunchProgram,Eye & Hand Tracker}"; Flags: nowait postinstall skipifsilent
