using System;
using System.ComponentModel;
using System.Windows.Input;

namespace ASD.CellUniverse.Infrastructure.Interfaces {

    public enum State { Started, Paused, Stopped }

    public interface IMainController : INotifyPropertyChanged {

        event Action Started, Paused, Resumed, Stopped, Reseted;

        event Action<State> StateChanged;

        State State { get; }

        ICommand Start { get; }
        ICommand Stop { get; }
    }
}