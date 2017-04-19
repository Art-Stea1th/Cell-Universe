using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Input;
using System.Windows.Threading;
using Prism.Commands;
using Prism.Mvvm;

namespace ASD.CellUniverse.ViewModels {

    internal sealed class ControlsPanelViewModel : BindableBase {

        private DispatcherTimer timer;

        private int currentResolutionIndex = 0;
        public List<Point> ResolutionsCollection { get; }

        public int CurrentResolutionIndex {
            get => currentResolutionIndex;
            set {
                SetProperty(ref currentResolutionIndex, value);
                RaisePropertyChanged(nameof(CellsHorizontally));
                RaisePropertyChanged(nameof(CellsVertically));
            }
        }

        public int CellsHorizontally => (int)ResolutionsCollection[CurrentResolutionIndex].X;
        public int CellsVertically => (int)ResolutionsCollection[CurrentResolutionIndex].Y;

        // ---
        private int ticksCount;
        public int TicksCount {
            get => ticksCount;
            set => SetProperty(ref ticksCount, value);
        }
        // ---

        public ICommand Start => new DelegateCommand(
            () => { timer.Start(); RaisePropertyChanged(); RaisePropertyChanged(nameof(Stop)); },
            () => !timer.IsEnabled);

        public ICommand Stop => new DelegateCommand(
            () => { timer.Stop(); RaisePropertyChanged(nameof(Start)); RaisePropertyChanged(); },
            () => timer.IsEnabled);

        public ControlsPanelViewModel() {
            ResolutionsCollection = new List<Point> {
                new Point(1280, 960), new Point(640, 480), new Point(320, 240)
            };

            timer = new DispatcherTimer() {
                //Interval = TimeSpan.FromMilliseconds(1000.0 / 30.0)
                Interval = TimeSpan.FromTicks(1) // speed-test :)
            };
            timer.Tick += (s, e) => TicksCount++;
        }
    }
}