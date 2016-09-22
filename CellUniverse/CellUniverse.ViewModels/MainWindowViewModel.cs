using System;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Threading;

namespace CellUniverse.ViewModels {

    using Infrastructure;
    using Infrastructure.Interfaces;
    using Models;
    using Models.Algorithms;

    public class MainWindowViewModel : ViewModelBase {

        private int width, height, layersCount;
        private int currentSpeed;

        private DispatcherTimer timer;
        private CellUniverseState currentState;

        private ICellUniverse multiverse;
        private Color[,] cellularData;

        private string startPauseButtonText = "Start";
        private string stopResetButtonText  = ". . .";

        public int MinSpeed { get; } = 10;
        public int MaxSpeed { get; } = 500;

        public MainWindowViewModel() {
            Initialize();
        }

        private void Initialize() {
            width = 266; height = 166; layersCount = 3;
            Speed = 10;
            Stop();
        }

        private void Start() {
            timer.Start();
            currentState = CellUniverseState.Started;
        }

        private void Pause() {
            timer.Stop();
            currentState = CellUniverseState.Paused;
        }

        private void Stop() {
            timer.Stop();
            currentState = CellUniverseState.Stopped;
            ConstructUniverse(width, height, layersCount);
        }

        private void ConstructUniverse(int width, int height, int layersCount) {
            multiverse   = new Multiverse(width, height, layersCount, new TheGameOfLifeClassic());
            CellularData = new Color[height, width];
        }

        public int Speed {
            get {
                return currentSpeed = IntValueLimiter(currentSpeed, MinSpeed, MaxSpeed);
            }
            set {
                currentSpeed = IntValueLimiter(value, MinSpeed, MaxSpeed);
                SetTimerInterval(TimeSpan.FromMilliseconds(value));
                OnPropertyChanged(GetMemberName((MainWindowViewModel c) => c.Speed));
            }
        }

        private void SetTimerInterval(TimeSpan interval) {
            if (timer == null) {
                timer = new DispatcherTimer();
                timer.Tick += (s, e) => { Update(); };
            }
            timer.Interval = interval;
        }

        private void Update() {
            CellularData = multiverse.GetNext();
        }

        public Color[,] CellularData {
            get {
                return cellularData;
            }
            set {
                cellularData = value;
                OnPropertyChanged(GetMemberName((MainWindowViewModel c) => c.CellularData));
            }
        }

        public string StartPauseResumeButtonText {
            get {
                return startPauseButtonText;
            }
            set {
                startPauseButtonText = value;
                OnPropertyChanged(GetMemberName((MainWindowViewModel c) => c.StartPauseResumeButtonText));
            }
        }

        public string StopResetButtonText {
            get {
                return stopResetButtonText;
            }
            set {
                stopResetButtonText = value;
                OnPropertyChanged(GetMemberName((MainWindowViewModel c) => c.StopResetButtonText));
            }
        }

        private int IntValueLimiter(int value, int minValue, int maxValue) {
            value = value < minValue ? minValue : value;
            value = value > maxValue ? maxValue : value;
            return value;
        }

        private RelayCommand _startPauseResumeSimulationCommand;
        private RelayCommand _stopResetSimulationCommand;

        public ICommand StartPauseResumeSimulation {
            get {
                return
                    (_startPauseResumeSimulationCommand) ??
                    (_startPauseResumeSimulationCommand =
                    new RelayCommand(ExecuteStartPauseResumeSimulationCommand, null));
            }
        }

        public ICommand StopResetSimulation {
            get {
                return
                    (_stopResetSimulationCommand) ??
                    (_stopResetSimulationCommand =
                    new RelayCommand(ExecuteStopResetSimulationCommand, CanExecuteStopResetSimulationCommand));
            }
        }

        public void ExecuteStartPauseResumeSimulationCommand(object parameter) {
            if (currentState == CellUniverseState.Started) {
                StartPauseResumeButtonText = "Resume";
                StopResetButtonText        = "Reset";
                Pause();
                return;
            }
            StartPauseResumeButtonText = "Pause";
            StopResetButtonText        = "Stop";
            Start();
        }

        public void ExecuteStopResetSimulationCommand(object parameter) {
            StartPauseResumeButtonText = "Start";
            StopResetButtonText        = ". . .";
            Stop();
        }

        public bool CanExecuteStopResetSimulationCommand(object parameter) {
            return
                currentState == CellUniverseState.Started ||
                currentState == CellUniverseState.Paused ? true : false;
        }        
    }
}