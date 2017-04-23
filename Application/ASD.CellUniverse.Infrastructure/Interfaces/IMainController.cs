using System;
using System.ComponentModel;
using System.Windows.Input;

namespace ASD.CellUniverse.Infrastructure.Interfaces {

    public interface IMainController : INotifyPropertyChanged {

        event Action Started, Paused, Stopped;

        ICommand Play { get; }
        ICommand Pause { get; }
        ICommand Stop { get; }
    }
}