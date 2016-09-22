using System;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Controls;
using System.Windows.Threading;

namespace CellUniverse.ViewModels {

    using Infrastructure;
    using Infrastructure.Interfaces;
    using Models;
    
    public class MainWindowViewModel : ViewModelBase {

        private ICellUniverse multiverse;
        private DispatcherTimer timer;
        private CellUniverseState currentState { get; set; } = CellUniverseState.Stopped;

        private Color[,] cellularData;

        public Color[,] CellularData {
            get {
                return cellularData;
            }
            set {
                cellularData = value;
                OnPropertyChanged(GetMemberName((MainWindowViewModel c) => c.CellularData));
            }
        }

        public MainWindowViewModel() {
            Initialize();
        }

        private void Initialize() {
            currentState = CellUniverseState.Stopped;
            Speed = 10;
            StartPauseResumeButtonText = "Start";
            StopResetButtonText = ". . .";
            multiverse = new Multiverse(266, 166, 3);
            timer = new DispatcherTimer();
            timer.Interval = TimeSpan.FromMilliseconds(Speed);
            timer.Tick += (s, e) => { Update(); };
            cellularData = new Color[166, 266];
        }

        internal void Start() {
            timer.Start();
            currentState = CellUniverseState.Started;
        }

        internal void Pause() {
            timer.Stop();
            currentState = CellUniverseState.Paused;
        }

        internal void Stop() {
            timer.Stop();
            currentState = CellUniverseState.Stopped;
            multiverse = new Multiverse(266, 166, 3);
            CellularData = new Color[166, 266];
        }

        internal void SetTimerInterval(TimeSpan interval) {
            timer.Interval = interval;
        }

        private void Update() {
            CellularData = multiverse.GetNext();
        }

        #region impl. ControlsViewModel

        private int _currentSpeed;

        private string _startPauseButtonText;
        private string _stopResetButtonText;

        public int MinSpeed { get; } = 10;
        public int MaxSpeed { get; } = 500;

        public int Speed {
            get {
                return _currentSpeed = IntValueLimiter(_currentSpeed, MinSpeed, MaxSpeed);
            }
            set {
                _currentSpeed = IntValueLimiter(value, MinSpeed, MaxSpeed);
                if (timer != null) {
                    timer.Interval = TimeSpan.FromMilliseconds(value);
                }                
                OnPropertyChanged(GetMemberName((MainWindowViewModel c) => c.Speed));
            }
        }

        public string StartPauseResumeButtonText {
            get {
                return _startPauseButtonText;
            }
            set {
                _startPauseButtonText = value;
                OnPropertyChanged(GetMemberName((MainWindowViewModel c) => c.StartPauseResumeButtonText));
            }
        }

        public string StopResetButtonText {
            get {
                return _stopResetButtonText;
            }
            set {
                _stopResetButtonText = value;
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

        public void ExecuteStartPauseResumeSimulationCommand(object parameter) {
            if (currentState == CellUniverseState.Started) {
                StartPauseResumeButtonText = "Resume";
                StopResetButtonText = "Reset";
                Pause();
                return;
            }
            StartPauseResumeButtonText = "Pause";
            StopResetButtonText = "Stop";
            Start();
        }

        public ICommand StopResetSimulation {
            get {
                return
                    (_stopResetSimulationCommand) ??
                    (_stopResetSimulationCommand =
                    new RelayCommand(ExecuteStopResetSimulationCommand, CanExecuteStopResetSimulationCommand));
            }
        }

        public void ExecuteStopResetSimulationCommand(object parameter) {
            StartPauseResumeButtonText = "Start";
            StopResetButtonText = ". . .";
            Stop();            
        }

        public bool CanExecuteStopResetSimulationCommand(object parameter) {
            return
                currentState == CellUniverseState.Started ||
                currentState == CellUniverseState.Paused ? true : false;
        }

        #endregion
    }
}