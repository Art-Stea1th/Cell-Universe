using System;
using System.Windows.Input;
using System.Windows.Controls;

namespace CellUniverse.ViewModels {

    using Infrastructure;

    public class MainWindowViewModel : ViewModelBase {

        private BinaryViewModel _binaryViewModel;

        public MainWindowViewModel(Canvas cellUniverseViewport) {
            _binaryViewModel = new BinaryViewModel(cellUniverseViewport);
            Speed = 10;
            StartPauseResumeButtonText = "Start";
            StopResetButtonText = ". . .";
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
                _binaryViewModel.SetTimerInterval(TimeSpan.FromMilliseconds(value));
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

        internal void CellSurfaceInvalidate() {
            _binaryViewModel.InvalidateView();
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
            if (_binaryViewModel.CurrentState == CellUniverseState.Started) {
                StartPauseResumeButtonText = "Resume";
                StopResetButtonText = "Reset";
                _binaryViewModel.Pause();
                return;
            }
            StartPauseResumeButtonText = "Pause";
            StopResetButtonText = "Stop";
            _binaryViewModel.Start();
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
            _binaryViewModel.Stop();
            Speed = _currentSpeed;
        }

        public bool CanExecuteStopResetSimulationCommand(object parameter) {
            return
                _binaryViewModel.CurrentState == CellUniverseState.Started ||
                _binaryViewModel.CurrentState == CellUniverseState.Paused ? true : false;
        }

        #endregion
    }
}