using System;
using System.Diagnostics;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Threading;


namespace CellUniverse.ViewModels {

    using Infrastructure;
    using Infrastructure.Interfaces;
    using Models;
    
    public class MainWindowViewModel : ViewModelBase {

        private int width, height, layersCount;
        private int delay, nextGenerationsTime;
        private int generationsPerSecond, framesPerSecond;
        private int totalGenerations;

        private DispatcherTimer timer;
        private Stopwatch stopwatch, totalStopwatch;
        private TimeSpan totalTime;
        private CellUniverseState currentState;

        private ICellUniverse multiverse;
        private Color[,] cellularData;

        private string startPauseButtonText = "Start";
        private string stopResetButtonText  = ". . .";

        public int MinSpeed { get; } = 0;
        public int MaxSpeed { get; } = 500;

        public int SpacingBetweenCells { get; set; }

        public MainWindowViewModel() {
            Initialize();
        }

        private void Initialize() {
            //width = 960; height = 540; layersCount = 3; SpacingBetweenCells = 1;
            width = 320; height = 180; layersCount = 3; SpacingBetweenCells = 1;
            Delay = 0;

            timer = new DispatcherTimer();
            timer.Tick += (s, e) => { Update(); };
            timer.Interval = TimeSpan.FromMilliseconds(Delay);

            stopwatch = new Stopwatch();
            totalStopwatch = new Stopwatch();

            Stop();
        }

        private void Start() {
            totalStopwatch.Start();
            timer.Start();
            currentState = CellUniverseState.Started;
        }

        private void Pause() {
            totalStopwatch.Stop();
            stopwatch.Stop();
            timer.Stop();
            currentState = CellUniverseState.Paused;            
        }

        private void Stop() {
            totalStopwatch.Stop();
            timer.Stop();
            currentState = CellUniverseState.Stopped;
            ConstructUniverse(width, height, layersCount);
            ResetCounters();
        }

        private void ConstructUniverse(int width, int height, int layersCount) {
            multiverse   = new Multiverse(width, height, layersCount);
            CellularData = new Color[height, width];
        }

        private void SetTimerInterval(TimeSpan interval) {
            if (timer != null) {
                timer.Interval = interval;
            }            
        }

        private void Update() {
            foreach (var layer in multiverse.GetNext()) {
                CellularData = layer;
            }
            UpdateCounters();
            stopwatch.Restart();
        }

        private void UpdateCounters() {

            //if (totalStopwatch.Elapsed >= TimeSpan.FromSeconds(30)) { // <--- !! for speed-test
            //    startPauseResumeSimulationCommand.Execute(null);
            //}

            TotalTime = totalStopwatch.Elapsed;
            TotalGenerations += layersCount;
            NextGenerationsTime = (int)stopwatch.ElapsedMilliseconds > 0 ? (int)stopwatch.ElapsedMilliseconds : 1;
            FramesPerSecond = (1000 / NextGenerationsTime);
            GenerationsPerSecond = FramesPerSecond * layersCount;            
        }

        private void ResetCounters() {
            TotalTime = TimeSpan.Zero;
            TotalGenerations = 0;
            FramesPerSecond = 0;
            GenerationsPerSecond = 0;
            NextGenerationsTime = 0;

            stopwatch.Reset();
            totalStopwatch.Reset();
        }

        public TimeSpan TotalTime {
            get {
                return totalTime;
            }
            set {
                totalTime = value;
                OnPropertyChanged(GetMemberName((MainWindowViewModel c) => c.TotalTime));
            }
        }

        public int TotalGenerations {
            get {
                return totalGenerations;
            }
            set {
                totalGenerations = value;
                OnPropertyChanged(GetMemberName((MainWindowViewModel c) => c.TotalGenerations));
            }
        }

        public int FramesPerSecond {
            get {
                return framesPerSecond;
            }
            set {
                framesPerSecond = value;
                OnPropertyChanged(GetMemberName((MainWindowViewModel c) => c.FramesPerSecond));
            }
        }

        public int GenerationsPerSecond {
            get {
                return generationsPerSecond;
            }
            set {
                generationsPerSecond = value;
                OnPropertyChanged(GetMemberName((MainWindowViewModel c) => c.GenerationsPerSecond));
            }
        }

        public int LayersCount {
            get {
                return layersCount;
            }
            set {
                layersCount = value;
                OnPropertyChanged(GetMemberName((MainWindowViewModel c) => c.LayersCount));
            }
        }

        public int NextGenerationsTime {
            get {
                return nextGenerationsTime;
            }
            set {
                nextGenerationsTime = value;
                OnPropertyChanged(GetMemberName((MainWindowViewModel c) => c.NextGenerationsTime));
            }
        }

        public int Delay {
            get {
                return delay = IntValueLimiter(delay, MinSpeed, MaxSpeed);
            }
            set {
                delay = IntValueLimiter(value, MinSpeed, MaxSpeed);
                SetTimerInterval(TimeSpan.FromMilliseconds(value));
                OnPropertyChanged(GetMemberName((MainWindowViewModel c) => c.Delay));
            }
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

        private RelayCommand startPauseResumeSimulationCommand;
        private RelayCommand stopResetSimulationCommand;

        public ICommand StartPauseResumeSimulation {
            get {
                return
                    (startPauseResumeSimulationCommand) ??
                    (startPauseResumeSimulationCommand =
                    new RelayCommand(ExecuteStartPauseResumeSimulationCommand, null));
            }
        }

        public ICommand StopResetSimulation {
            get {
                return
                    (stopResetSimulationCommand) ??
                    (stopResetSimulationCommand =
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