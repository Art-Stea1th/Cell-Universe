using System;
using System.ComponentModel;
using System.Windows.Input;

namespace ASD.CellUniverse.Infrastructure.Interfaces {

    public enum State { Started, Paused, Stopped }

    internal interface IGenerationController : INotifyPropertyChanged {

        State State { get; }

        event Action<State> StateChanged;

        event Action Started, Paused, Resumed, Stopped, Reseted;

        ICommand Start { get; }
        ICommand Stop { get; }
    }
}