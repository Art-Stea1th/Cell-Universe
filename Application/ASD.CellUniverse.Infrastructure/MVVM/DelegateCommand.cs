using System;
using System.Windows.Input;

namespace ASD.CellUniverse.Infrastructure.MVVM {

    public class DelegateCommand : ICommand {

        private readonly Action<object> _execute;
        private readonly Predicate<object> _canExecute;

        public event EventHandler CanExecuteChanged {
            add => CommandManager.RequerySuggested += value;
            remove => CommandManager.RequerySuggested -= value;
        }

        public DelegateCommand(Action<object> execute, Predicate<object> canExecute = null) {
            _execute = execute ?? throw new ArgumentNullException("execute");
            _canExecute = canExecute;
        }

        public bool CanExecute(object parameter)
            => _canExecute == null ? true : _canExecute.Invoke(parameter);

        public void Execute(object parameter) => _execute.Invoke(parameter);
    }
}