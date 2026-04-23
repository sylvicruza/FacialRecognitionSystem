#define MyAppName "KairosTrack"
#ifndef MyAppVersion
#define MyAppVersion "1.0.0"
#endif
#define MyAppPublisher "SlyApps"
#define MyAppExeName "KairosTrack.exe"
#define MyAppFolder "KairosTrack"

[Setup]
AppId={{5E471D31-85B6-4C43-BB6C-1A72F3625A20}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL=https://github.com/sylvicruza/FacialRecognitionSystem
AppSupportURL=https://github.com/sylvicruza/FacialRecognitionSystem/issues
AppUpdatesURL=https://github.com/sylvicruza/FacialRecognitionSystem/releases/latest
DefaultDirName={autopf}\{#MyAppFolder}
DefaultGroupName={#MyAppName}
DisableDirPage=no
DisableProgramGroupPage=yes
OutputDir=installer_output
OutputBaseFilename=KairosTrackSetup-{#MyAppVersion}
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
CloseApplications=yes
CloseApplicationsFilter={#MyAppExeName}
SetupIconFile=static\images\kairostrack-icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
VersionInfoVersion={#MyAppVersion}
VersionInfoCompany={#MyAppPublisher}
VersionInfoDescription=KairosTrack desktop attendance client
VersionInfoProductName={#MyAppName}
AppMutex=KairosTrackDesktopMutex

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
Source: "dist\KairosTrack.exe"; DestDir: "{app}"; DestName: "{#MyAppExeName}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{localappdata}\KairosTrack"
Type: filesandordirs; Name: "{userappdata}\KairosTrack"
Type: filesandordirs; Name: "{commonappdata}\KairosTrack"
Type: filesandordirs; Name: "{tmp}\KairosTrack"
