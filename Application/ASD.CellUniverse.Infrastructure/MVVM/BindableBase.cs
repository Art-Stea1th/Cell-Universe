using System;
using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace ASD.CellUniverse.Infrastructure.MVVM {

    public abstract class BindableBase : INotifyPropertyChanged, IDisposable {

        public event PropertyChangedEventHandler PropertyChanged;

        protected BindableBase() { }

        protected void Set<T>(ref T oldValue, T newValue, [CallerMemberName] string propertyName = null) {
            oldValue = newValue; RaisePropertyChanged(propertyName);
        }

        protected void RaisePropertyChanged([CallerMemberName] string propertyName = null)
            => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));

        public void Dispose() => OnDispose();

        protected virtual void OnDispose() { }
    }
}