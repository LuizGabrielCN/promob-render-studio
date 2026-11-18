; Promob Render Studio — instalador por usuário (Inno Setup 6).
; Compilado pelo packaging\windows\build.ps1:
;   ISCC /DAppVersion=0.2.0 /DStageDir=<pacote montado> /DOutputDir=<saída> installer.iss
; Não requer administrador: instala em %LOCALAPPDATA%\Programs\Promob Render Studio.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef StageDir
  #error StageDir não informado (use build.ps1)
#endif
#ifndef OutputDir
  #define OutputDir "."
#endif

#define AppName "Promob Render Studio"
#define AppUrl "https://github.com/LuizGabrielCN/promob-render-studio"
#define DataDirName "PromobRenderStudio"

[Setup]
AppId={{807EF539-A9F9-4019-8081-FD705918FCF6}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=Promob Render Studio (código aberto, MIT)
AppPublisherURL={#AppUrl}
AppSupportURL={#AppUrl}/issues
AppUpdatesURL={#AppUrl}/releases
VersionInfoVersion={#AppVersion}
DefaultDirName={localappdata}\Programs\{#AppName}
DisableDirPage=auto
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir={#OutputDir}
OutputBaseFilename=PromobRenderStudio-Setup
SetupIconFile={#StageDir}\app\promob-render-studio.ico
UninstallDisplayIcon={app}\app\promob-render-studio.ico
UninstallDisplayName={#AppName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "brazilianportuguese"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"

[Tasks]
Name: "desktopicon"; Description: "Criar atalho na área de trabalho"; GroupDescription: "Atalhos:"; Flags: unchecked

[InstallDelete]
; Atualizações substituem o Python privado e o app por completo (evita arquivos antigos).
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\app"
Type: filesandordirs; Name: "{app}\runtime"

[Files]
Source: "{#StageDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\app\promob_launcher.py"""; WorkingDir: "{app}\app"; IconFilename: "{app}\app\promob-render-studio.ico"; Comment: "Abre o Promob Render Studio no navegador"
Name: "{autoprograms}\Encerrar {#AppName}"; Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\app\promob_launcher.py"" --stop"; WorkingDir: "{app}\app"; IconFilename: "{app}\app\promob-render-studio.ico"; Comment: "Encerra o servidor local do Promob Render Studio"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\app\promob_launcher.py"""; WorkingDir: "{app}\app"; IconFilename: "{app}\app\promob-render-studio.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\app\promob_launcher.py"""; WorkingDir: "{app}\app"; Description: "Abrir o {#AppName} agora"; Flags: postinstall nowait skipifsilent

[UninstallRun]
Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\app\promob_launcher.py"" --stop"; WorkingDir: "{app}\app"; Flags: runhidden waituntilterminated; RunOnceId: "StopServer"

[UninstallDelete]
; Inclui as bibliotecas de IA instaladas sob demanda dentro do Python privado.
Type: filesandordirs; Name: "{app}"
Type: filesandordirs; Name: "{localappdata}\{#DataDirName}\cache"
Type: filesandordirs; Name: "{localappdata}\{#DataDirName}\logs"
Type: filesandordirs; Name: "{localappdata}\{#DataDirName}\uv-cache"
Type: files; Name: "{localappdata}\{#DataDirName}\instancia.json"
Type: files; Name: "{localappdata}\{#DataDirName}\instancia.lock"
Type: dirifempty; Name: "{localappdata}\{#DataDirName}"

[Code]
function StopRunningApp(): Boolean;
var
  ResultCode: Integer;
  Launcher: String;
begin
  Result := True;
  Launcher := ExpandConstant('{app}\app\promob_launcher.py');
  if FileExists(Launcher) and FileExists(ExpandConstant('{app}\python\pythonw.exe')) then
    Exec(ExpandConstant('{app}\python\pythonw.exe'), '"' + Launcher + '" --stop', ExpandConstant('{app}\app'),
         SW_HIDE, ewWaitUntilTerminated, ResultCode);
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  StopRunningApp();
  Result := '';
end;

function HfHubDir(): String;
begin
  Result := GetEnv('HF_HUB_CACHE');
  if Result = '' then
  begin
    Result := GetEnv('HF_HOME');
    if Result = '' then
      Result := ExpandConstant('{%USERPROFILE}') + '\.cache\huggingface';
    Result := Result + '\hub';
  end;
end;

procedure DeleteModels();
var
  Hub: String;
begin
  DelTree(ExpandConstant('{localappdata}\{#DataDirName}\models'), True, True, True);
  Hub := HfHubDir();
  DelTree(Hub + '\models--othytenk--realesrgan-x4plus', True, True, True);
  DelTree(Hub + '\models--IDEA-Research--grounding-dino-tiny', True, True, True);
  DelTree(Hub + '\models--facebook--sam2.1-hiera-tiny', True, True, True);
  DelTree(Hub + '\models--depth-anything--Depth-Anything-V2-Small-hf', True, True, True);
  DelTree(Hub + '\models--lllyasviel--control_v11f1p_sd15_depth', True, True, True);
  DelTree(Hub + '\models--lllyasviel--control_v11p_sd15_canny', True, True, True);
  DelTree(Hub + '\models--lllyasviel--control_v11f1e_sd15_tile', True, True, True);
  DelTree(Hub + '\models--lllyasviel--ic-light', True, True, True);
  DelTree(Hub + '\models--emilianJR--epiCRealism', True, True, True);
  DelTree(Hub + '\models--ByteDance--Hyper-SD', True, True, True);
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
  begin
    if SuppressibleMsgBox(
      'Deseja apagar também os modelos de IA baixados pelo app (podem ocupar até 9 GB)?' + #13#10 + #13#10 +
      'Eles ficam em ' + ExpandConstant('{localappdata}\{#DataDirName}\models') + ' e no cache do Hugging Face (' + HfHubDir() + ').' + #13#10 +
      'Escolha "Não" se você usa esses modelos em outra instalação do Promob Render Studio.',
      mbConfirmation, MB_YESNO or MB_DEFBUTTON2, IDNO) = IDYES then
      DeleteModels();
    RemoveDir(ExpandConstant('{localappdata}\{#DataDirName}'));
  end;
end;
