using System;
using System.ComponentModel;
using System.Windows.Input;

namespace ASD.CellUniverse.Infrastructure.Interfaces {

    public interface IMainController : INotifyPropertyChanged {

        event Action Started, Paused, Resumed, Stopped, Reseted;

        ICommand PlayPauseResume { get; }
        ICommand StopReset { get; }
    }
}