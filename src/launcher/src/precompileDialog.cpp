#include "precompileDialog.h"

#include <QCheckBox>
#include <QFileInfo>
#include <QLabel>
#include <QProcess>
#include <QProgressBar>
#include <QPushButton>
#include <QRadioButton>
#include <QSettings>

#include <algorithm>

#include "ui_precompile_dialog.h"

namespace {
constexpr char KEY_MODE[]    = "mode";
constexpr char KEY_ENTRIES[] = "entries";
constexpr char KEY_ASK[]     = "ask";

constexpr int SAMPLE_COUNT     = 10;
constexpr int MIN_RATE_SAMPLES = 3;

QString PrecompileGroup(const QString& title_id) {
	return QStringLiteral("Precompile/") + title_id;
}

QString FormatEta(double seconds) {
	const auto total = static_cast<int>(seconds);
	return QStringLiteral("%1:%2")
	    .arg(total / 60, 2, 10, QLatin1Char('0'))
	    .arg(total % 60, 2, 10, QLatin1Char('0'));
}

// The value of a "NAME\tvalue" line the emulator prints for the launcher, or -1 for any other
// line. The report that follows SHADERS is prose and is ignored the same way.
int ReportValue(const QByteArray& line, const char* name) {
	const auto prefix = QByteArray(name) + '\t';
	if (!line.startsWith(prefix)) {
		return -1;
	}
	const auto field = line.mid(prefix.size());
	if (const auto tab = field.indexOf('\t'); tab >= 0) {
		return field.left(tab).toInt();
	}
	return field.toInt();
}
} // namespace

PrecompileDialog::Prefs PrecompileDialog::ReadPrefs(const QString& settings_file,
                                                    const QString& title_id) {
	Prefs prefs;
	if (settings_file.isEmpty() || title_id.isEmpty()) {
		return prefs;
	}

	QSettings settings(settings_file, QSettings::IniFormat);
	settings.beginGroup(PrecompileGroup(title_id));
	const auto mode =
	    settings.value(QString::fromLatin1(KEY_MODE), static_cast<int>(Mode::Auto)).toInt();
	if (mode >= static_cast<int>(Mode::Direct) && mode <= static_cast<int>(Mode::Manual)) {
		prefs.mode = static_cast<Mode>(mode);
	}
	prefs.entries = settings.value(QString::fromLatin1(KEY_ENTRIES), 0u).toUInt();
	prefs.ask     = settings.value(QString::fromLatin1(KEY_ASK), true).toBool();
	settings.endGroup();

	return prefs;
}

void PrecompileDialog::WritePrefs(const QString& settings_file, const QString& title_id,
                                  const Prefs& prefs) {
	if (settings_file.isEmpty() || title_id.isEmpty()) {
		return;
	}

	QSettings settings(settings_file, QSettings::IniFormat);
	settings.beginGroup(PrecompileGroup(title_id));
	settings.setValue(QString::fromLatin1(KEY_MODE), static_cast<int>(prefs.mode));
	settings.setValue(QString::fromLatin1(KEY_ENTRIES), prefs.entries);
	settings.setValue(QString::fromLatin1(KEY_ASK), prefs.ask);
	settings.endGroup();
}

PrecompileDialog::PrecompileDialog(const QString& game_name, const QString& title_id,
                                   const QString& interpreter, const QStringList& args,
                                   const QString& settings_file, bool unattended, QWidget* parent)
    : QDialog(parent), m_ui(new Ui::PrecompileDialog), m_title_id(title_id),
      m_settings_file(settings_file), m_interpreter(interpreter), m_args(args) {
	m_ui->setupUi(this);

	m_ui->label_game->setText(tr("Game: %1").arg(game_name));

	m_prefs = ReadPrefs(settings_file, title_id);

	switch (m_prefs.mode) {
		case Mode::Direct: m_ui->radio_direct->setChecked(true); break;
		case Mode::Manual: m_ui->radio_manual->setChecked(true); break;
		case Mode::Auto: m_ui->radio_auto->setChecked(true); break;
	}
	m_ui->check_dont_ask->setChecked(!m_prefs.ask);
	m_ui->button_launch->setVisible(false);
	m_ui->bar->setVisible(false);
	m_ui->label_progress->setVisible(false);

	connect(m_ui->button_start, &QPushButton::clicked, this, &PrecompileDialog::on_start);
	connect(m_ui->button_launch, &QPushButton::clicked, this, &PrecompileDialog::on_launch);
	connect(m_ui->button_cancel, &QPushButton::clicked, this, [this] { done(QDialog::Rejected); });

	connect(&m_eta_timer, &QTimer::timeout, this, &PrecompileDialog::UpdateEta);
	connect(&m_process, &QProcess::readyReadStandardOutput, this, &PrecompileDialog::ReadOutput);
	connect(&m_process, &QProcess::finished, this, &PrecompileDialog::on_finished);
	connect(&m_process, &QProcess::errorOccurred, this, &PrecompileDialog::on_error);

	m_ui->label_rate->setText(m_prefs.entries != 0
	                              ? tr("Last precompilation translated %1 shaders.")
	                                    .arg(m_prefs.entries)
	                              : tr("Every shader the game's files hold is translated up front."));

	if (unattended) {
		// "Do not ask again" is set for this title: hide the choices and go straight to the work.
		m_ui->group_mode->setVisible(false);
		m_ui->check_dont_ask->setVisible(false);
		on_start();
	}
}

PrecompileDialog::~PrecompileDialog() {
	StopPrecompile();
	delete m_ui;
}

PrecompileDialog::Mode PrecompileDialog::SelectedMode() const {
	if (m_ui->radio_direct->isChecked()) {
		return Mode::Direct;
	}
	if (m_ui->radio_manual->isChecked()) {
		return Mode::Manual;
	}
	return Mode::Auto;
}

void PrecompileDialog::on_start() {
	m_prefs.mode = SelectedMode();
	m_prefs.ask  = !m_ui->check_dont_ask->isChecked();
	WritePrefs(m_settings_file, m_title_id, m_prefs);

	m_mode = m_prefs.mode;
	if (m_mode == Mode::Direct) {
		accept();
		return;
	}

	StartPrecompile();
}

void PrecompileDialog::StartPrecompile() {
	m_ui->group_mode->setEnabled(false);
	m_ui->check_dont_ask->setEnabled(false);
	m_ui->button_start->setVisible(false);
	m_ui->bar->setVisible(true);
	m_ui->label_progress->setVisible(true);
	m_ui->label_rate->setVisible(true);
	m_ui->bar->setRange(0, 0); /* no denominator until DISTINCT arrives */
	m_ui->label_progress->setText(tr("Reading the game's shaders…"));
	m_ui->label_rate->setText(QString());

	m_pending.clear();
	m_total = 0;
	m_done  = 0;
	m_samples.clear();
	m_clock.start();
	m_eta_timer.start(1000);

	m_process.setProgram(m_interpreter);
	m_process.setArguments(m_args + QStringList {QStringLiteral("--precompile")});
	m_process.setWorkingDirectory(QFileInfo(m_interpreter).absolutePath());
	// No new console: the emulator's report is read here, and the window it would create is
	// hidden for this mode.
	m_process.start();
}

void PrecompileDialog::StopPrecompile() {
	m_eta_timer.stop();
	if (m_process.state() == QProcess::NotRunning) {
		return;
	}
	// Posts WM_CLOSE to the (hidden) window, so the emulator leaves through its own quit path.
	m_process.terminate();
	if (!m_process.waitForFinished(5000)) {
		m_process.kill();
		m_process.waitForFinished(2000);
	}
}

// One line of the report at a time. Everything else the emulator logs is left alone.
void PrecompileDialog::ReadOutput() {
	m_pending += m_process.readAllStandardOutput();
	int newline = 0;
	while ((newline = m_pending.indexOf('\n')) >= 0) {
		const auto line = m_pending.left(newline);
		m_pending.remove(0, newline + 1);

		if (const int total = ReportValue(line, "DISTINCT"); total >= 0) {
			m_total = static_cast<uint32_t>(std::max(total, 0));
			m_ui->bar->setRange(0, static_cast<int>(std::max(m_total, 1u)));
			m_ui->bar->setValue(0);
			continue;
		}
		if (const int done = ReportValue(line, "PROGRESS"); done >= 0) {
			m_done = static_cast<uint32_t>(done);
			continue;
		}
		if (const int seeded = ReportValue(line, "PRECOMPILED"); seeded >= 0) {
			// The last line of a successful run: the count the bar should end on.
			m_done = m_total;
			continue;
		}
	}
	UpdateEta();
}

void PrecompileDialog::UpdateEta() {
	const auto elapsed = m_clock.elapsed();

	m_samples.append({elapsed, m_done});
	while (m_samples.size() > SAMPLE_COUNT) {
		m_samples.removeFirst();
	}

	if (m_done != 0 && m_total != 0) {
		m_ui->bar->setValue(static_cast<int>(std::min(m_done, m_total)));
	}

	double rate = 0.0;
	if (m_samples.size() >= MIN_RATE_SAMPLES) {
		const auto& oldest  = m_samples.first();
		const auto  span_ms = static_cast<double>(elapsed - oldest.first);
		if (span_ms > 0.0 && m_samples.last().second > oldest.second) {
			rate = static_cast<double>(m_samples.last().second - oldest.second) * 1000.0 / span_ms;
		}
	}

	if (m_total == 0) {
		m_ui->label_progress->setText(tr("%1 shaders translated").arg(m_done));
	} else {
		const auto percent = std::min<uint32_t>(100u, m_done * 100u / m_total);

		QString eta = QStringLiteral("--:--");
		if (m_done >= m_total) {
			eta = QStringLiteral("0:00");
		} else if (rate > 0.0) {
			eta = FormatEta((static_cast<double>(m_total) - static_cast<double>(m_done)) / rate);
		}
		m_ui->label_progress->setText(tr("%1 / %2 — %3% — ETA %4")
		                                  .arg(QString::number(m_done), QString::number(m_total),
		                                       QString::number(percent), eta));
	}

	m_ui->label_rate->setText(rate > 0.0 ? tr("%1 shaders/s").arg(rate, 0, 'f', 1)
	                                     : tr("translating…"));
}

void PrecompileDialog::on_finished(int exit_code, QProcess::ExitStatus status) {
	m_eta_timer.stop();
	ReadOutput();
	UpdateEta();

	if (m_done > m_prefs.entries) {
		m_prefs.entries = m_done;
		WritePrefs(m_settings_file, m_title_id, m_prefs);
	}

	if (exit_code != 0 || status != QProcess::NormalExit) {
		m_ui->label_rate->setText(tr("Precompilation ended with code %1 (%2 shaders).")
		                              .arg(exit_code)
		                              .arg(m_done));
	} else if (m_mode == Mode::Auto) {
		accept();
		return;
	} else {
		m_ui->label_rate->setText(tr("Precompilation finished: %1 shaders.").arg(m_done));
	}

	// Manual mode, or a run that failed: keep the dialog and let the user decide.
	m_ui->button_launch->setVisible(true);
	m_ui->button_launch->setEnabled(true);
	m_ui->button_launch->setDefault(true);
}

void PrecompileDialog::on_error(QProcess::ProcessError error) {
	if (error != QProcess::FailedToStart) {
		return;
	}
	// No finished() follows a failed start: report it and fall back to a manual launch.
	m_eta_timer.stop();
	m_ui->label_rate->setText(
	    tr("The emulator could not be started: %1").arg(m_process.errorString()));
	m_ui->button_launch->setVisible(true);
	m_ui->button_launch->setEnabled(true);
	m_ui->button_launch->setDefault(true);
}

void PrecompileDialog::on_launch() {
	accept();
}

void PrecompileDialog::done(int result) {
	if (result == QDialog::Rejected) {
		StopPrecompile();
	}
	QDialog::done(result);
}
